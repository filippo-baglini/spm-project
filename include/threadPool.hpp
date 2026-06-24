#pragma once

#include <vector>
#include <deque>
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <algorithm>
#include <stdexcept> 

#include "taskFactory.hpp"
#include "Affinity.hpp"

class threadPool {
#if !defined(__cpp_lib_move_only_function) || __cpp_lib_move_only_function < 202110
    using Task = std::function<void()>;
#else
    using Task = std::move_only_function<void()>;
#endif
	static constexpr std::chrono::microseconds WAIT_MICROSECONDS{200};
	
    std::vector<std::jthread> workers_;   // Worker threads owned by the pool
	std::vector<unsigned> affinity_;      // Optional list of logical CPUs used to pin Workers
    std::mutex m_;                        // protects shared state
    std::condition_variable cv_;          // used to wake sleeping workers
	std::condition_variable cv_wait_;     // Used to sleep until there are no inflight_ tasks
    bool done_ = false;                   // becomes true when shutdown starts
    std::deque<Task> q_;                  // queue of pending tasks
	std::size_t inflight_{0};             // Number of submitted tasks that have not completed yet

    void worker_loop() {
        for (;;) {
            Task task;  // local task extracted from the shared queue

            {
                std::unique_lock<std::mutex> lk(m_);

                // Wait until:
                // 1. shutdown has been requested, or
                // 2. there is at least one task to execute
                cv_.wait(lk, [&] { return done_ || !q_.empty(); });

                // If shutdown was requested and no work is left, terminate this Worker
                if (done_ && q_.empty()) return;

                // Remove one task from the front of the queue.
                task = std::move(q_.front());
                q_.pop_front();
            }

            // Execute the task outside the critical section.
            task();

			// The task has completed. Update the global number of in-flight tasks
			// If this was the last one, wake threads waiting for global completion
			dec_inflight_and_notify();
        }
    }

	// Decrement the number of in-flight tasks under the same mutex used by
	// wait_completion().
	// A Worker could set inflight_ to 0 and notify just before the waiting
	// thread go to sleep (i.e., enters cv_wait_.wait()), causing a lost wake-up
	void dec_inflight_and_notify() noexcept {
		std::unique_lock<std::mutex> lk(m_);
		if (--inflight_ == 0) {
			lk.unlock();
			cv_wait_.notify_all();
		}
	}

    // Try to steal one pending task from the shared queue and execute it
    // Returns true if a task was found and executed, false otherwise
    bool try_execute_one_pending_task() {
        Task task;
        {
            std::lock_guard<std::mutex> lk(m_);
            if (q_.empty())
                return false;

            task = std::move(q_.front());
            q_.pop_front();
        }

        // The calling thread temporarily behaves like an extra worker
        task();

        // The task has completed, so update the number of in-flight tasks
        dec_inflight_and_notify();
        return true;
    }

public:
    explicit threadPool(unsigned n = std::max(1u, std::thread::hardware_concurrency()),
						std::string affinity={}) {
        // hardware_concurrency() may return 0 on some systems
        if (n == 0) {
            throw std::invalid_argument("threadPool: n must be > 0");
        }

        workers_.reserve(n);

		affinity_ = affinity::parse_cpu_list(affinity);
		
        // Start n worker threads.
        for (unsigned i = 0; i < n; ++i) {
			// Choose CPU for this worker if provided (wrap-around if fewer CPUs)
            int cpu = -1;
            if (!affinity_.empty())
                cpu = static_cast<int>(affinity_[i % affinity_.size()]);
			
			workers_.emplace_back([this, cpu, i, AFF=affinity::get_current_cpu]{
				if (cpu>=0) affinity::pin_thread_to_core(cpu);

				//std::printf("Worker %u running on CPU %u\n", i, AFF());

				worker_loop();
			});
        }
    }

	//
	// The pool is neither copyable nor movable.
	// Copying is ill-defined because the pool owns threads and synchronization state.
	// Moving is also unsafe here because worker threads capture 'this' and are therefore
	// tied to the identity and address of the current object.
	//
    threadPool(const threadPool&)            = delete;
    threadPool& operator=(const threadPool&) = delete;
    threadPool(threadPool&&)                 = delete;
    threadPool& operator=(threadPool&&)      = delete;

    ~threadPool() {
		// Graceful shutdown:
		// 1. wait until no submitted task is still pending or running
		// 2. request worker termination
		// 3. wake sleeping workers
		// 4. join all threads
		//
		// Note: if running tasks keep submitting new work to this pool,
		// destruction may be delayed until that activity stops
		wait_completion();
		
        {
            std::lock_guard<std::mutex> lk(m_);
            done_ = true;   // signal shutdown
        }

        // Wake all workers so they can either:
        // - finish pending tasks, or
        // - exit if the queue is empty
        cv_.notify_all();

        // Wait for the termination of all workers.
        // Note: std::jthread would also join automatically in its destructor,
        // but keeping the explicit join makes shutdown behavior very clear.
        for (auto& thread : workers_) {
            thread.join();
        }
    }

    template<class F, class... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>> 
    {
        static_assert(std::is_invocable_v<F, Args...>, "F(Args...) is not callable");

        // Convert the user callable plus its arguments into:
        // - a uniform runnable task with signature void()
        // - the future associated with its result
        auto [task, future] = make_task(std::forward<F>(f), std::forward<Args>(args)...);

        {
            std::lock_guard<std::mutex> lk(m_);

            // Once destruction has started, the pool no longer accepts new work.
            if (done_) {
                throw std::runtime_error("threadPool: submit() called during shutdown");
            }

            q_.emplace_back(std::move(task));
			++inflight_;
        }

        // Wake one worker to execute the newly submitted task
        cv_.notify_one();

        return future;
    }

	// Wait until the pool becomes idle, i.e., no submitted task is still pending or running
	void wait_completion() {
		std::unique_lock<std::mutex> lk(m_);
		cv_wait_.wait(lk, [&] { return inflight_ == 0; });
	}

    // Cooperative helping wait:
    // while the target future is not ready, the caller tries to help the pool
    // by executing pending tasks instead of remaining completely blocked.
    //
    // This is useful when tasks recursively generate other tasks, or when the
    // waiting thread can contribute useful work and reduce overall idle time.
    template<class T>
    T wait_future(std::future<T>& fut) {
        using namespace std::chrono_literals;

        for (;;) {
            if (fut.wait_for(0s) == std::future_status::ready)
                break;

            if (try_execute_one_pending_task())
                continue; // one task executed, keep going

			// No task is currently available to help with
			// Sleep only for a short time, then recheck both the queue and the future.
			// The timeout is a responsiveness trade-off, not a correctness requirement
			std::unique_lock<std::mutex> lk(m_);
            cv_.wait_for(lk, WAIT_MICROSECONDS, [&] {
                return done_ || !q_.empty();
            });
        }

        return fut.get();
    }

    // Overload for future<void>
    void wait_future(std::future<void>& fut) {
        using namespace std::chrono_literals;

        for (;;) {
            if (fut.wait_for(0s) == std::future_status::ready)
                break;

            if (try_execute_one_pending_task())
                continue;

            std::unique_lock<std::mutex> lk(m_);
            cv_.wait_for(lk, WAIT_MICROSECONDS, [&] {
                return done_ || !q_.empty();
            });
        }

        fut.get();
    }
};
