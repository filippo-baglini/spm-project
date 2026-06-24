#pragma once

#include <future>
#include <functional>
#include <tuple>
#include <type_traits>
#include <memory>   // needed in the C++20 branch for shared_ptr / make_shared

#if !defined(__cpp_lib_move_only_function) || __cpp_lib_move_only_function < 202110

// ============================================================================
// C++20 version
// ============================================================================
// In C++20 we do not have std::move_only_function, so we use std::function<void()>
// as the uniform runnable task type.
//
// Problem:
// - std::packaged_task is move-only
// - std::function requires its target to be copy-constructible
//
// Solution:
// - allocate the packaged_task on the heap
// - manage it through std::shared_ptr
// - capture that shared_ptr inside a lambda, which then becomes copyable
// ============================================================================

template<class F, class... Args>
auto make_task(F&& f, Args&&... args)
    -> std::pair<
        std::function<void()>,
        std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
    >
{
    // Decayed type of the callable.
    // std::decay_t removes references and cv-qualifiers and performs the usual
    // array/function to pointer adjustments when needed.
    using Fn = std::decay_t<F>;

    // Store the arguments inside a tuple, again in decayed form.
    // This means that arguments are stored by value by default, either copied
    // or moved into the tuple.
    //
    // NOTE: If reference semantics are desired, the caller must explicitly use
    // std::ref(...) or std::cref(...).
    using Tp = std::tuple<std::decay_t<Args>...>;

    // Return type of invoking the callable with the stored argument types.
    using R = std::invoke_result_t<Fn, std::decay_t<Args>...>;

    // Build a packaged_task<R()>.
    //
    // The key idea is that we "bind" the callable and its arguments together
    // into a nullary task, that is, a task with signature R().
    //
    // Later, when this task is executed, it will perform fn(args...).
    // The packaged_task will automatically store either:
    // - the returned value, or
    // - any thrown exception
    // into its associated shared state, which is observed through a future.
    auto pkg = std::make_shared<std::packaged_task<R()>>(
        [fn = Fn(std::forward<F>(f)),
         tp = Tp(std::forward<Args>(args)...)]() mutable -> R
        {
            // std::apply expands the tuple 'tp' and invokes 'fn' with those arguments.
            //
            // We use std::move(fn) and std::move(tp) because this task is intended
            // to be executed only once. After invocation, the stored callable and
            // arguments are no longer needed.
            if constexpr (std::is_void_v<R>) {
                std::apply(std::move(fn), std::move(tp));
            } else {
                return std::apply(std::move(fn), std::move(tp));
            }
        }
    );

    // Obtain the future associated with the packaged_task.
    // The future will later provide access to:
    // - the return value, or
    // - the exception thrown during execution.
    auto fut = pkg->get_future();

    // Create a uniform runnable task with signature void().
    // Executing this wrapper simply invokes the underlying packaged_task.
    //
    // Note that calling this task is still synchronous with respect to the
    // calling thread. Asynchronous execution only happens if this task is
    // submitted to another thread or to a thread pool.
    std::function<void()> task =
        [pkg = std::move(pkg)]() mutable {
            (*pkg)();
        };

    // Return:
    // - the uniform runnable task
    // - the future associated with its result
    return { std::move(task), std::move(fut) };
}

#else

// ============================================================================
// C++23 version
// ============================================================================
// In C++23 we can use std::move_only_function<void()>.
//
// Advantage:
// - the wrapped callable no longer needs to be copyable
// - therefore we can move a std::packaged_task directly into the task wrapper
// - no heap allocation and no shared_ptr are needed
// ============================================================================

template<class F, class... Args>
auto make_task(F&& f, Args&&... args)
    -> std::pair<
        std::move_only_function<void()>,
        std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
    >
{
    using Fn = std::decay_t<F>;
    using Tp = std::tuple<std::decay_t<Args>...>;
    using R  = std::invoke_result_t<Fn, std::decay_t<Args>...>;

    // Create a packaged_task<R()> that captures:
    // - the callable
    // - the arguments already bound inside a tuple
    //
    // The result of the computation will later be available through the future.
    std::packaged_task<R()> pt(
        [fn = Fn(std::forward<F>(f)),
         tp = Tp(std::forward<Args>(args)...)]() mutable -> R
        {
            if constexpr (std::is_void_v<R>) {
                std::apply(std::move(fn), std::move(tp));
            } else {
                return std::apply(std::move(fn), std::move(tp));
            }
        }
    );

    // Retrieve the future associated with the packaged_task.
    auto fut = pt.get_future();

    // Thanks to std::move_only_function, we can move the packaged_task directly
    // into a uniform wrapper with signature void().
    //
    // This gives us a generic runnable task that can be stored in a queue and
    // executed later, while the result is observed through the future.
    std::move_only_function<void()> task = std::move(pt);

    return { std::move(task), std::move(fut) };
}

#endif
