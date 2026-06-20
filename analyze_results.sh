#!/bin/bash
# Analyze SpMV benchmark results using standard Unix tools
# No Python required - works on any cluster

results_dir="results"

compare_dumps() {
    local seq_dump="$1"
    local thread_dump="$2"

    if [ ! -f "$seq_dump" ]; then
        echo "  seq dump missing: $seq_dump"
        return
    fi
    if [ ! -f "$thread_dump" ]; then
        echo "  thread dump missing: $thread_dump"
        return
    fi

    if cmp -s "$seq_dump" "$thread_dump"; then
        echo "  dump match: identical"
        return
    fi

    local seq_lines th_lines
    seq_lines=$(wc -l < "$seq_dump")
    th_lines=$(wc -l < "$thread_dump")
    if [ "$seq_lines" -ne "$th_lines" ]; then
        echo "  dump mismatch: line count differs ($seq_lines vs $th_lines)"
        return
    fi

    local maxdiff
    maxdiff=$(paste "$seq_dump" "$thread_dump" | awk 'BEGIN {max=0} {diff=$1-$2; if(diff<0) diff=-diff; if(diff>max) max=diff} END {printf "%.17g", max}')
    echo "  dump mismatch: max abs diff=$maxdiff"
}

if [ ! -d "$results_dir" ]; then
    echo "Error: 'results' directory not found."
    echo "Run sbatch script first to generate results."
    exit 1
fi

# Check if there are any result files
if [ $(ls -1 "$results_dir"/*.txt 2>/dev/null | wc -l) -eq 0 ]; then
    echo "Error: No result files found in 'results' directory."
    exit 1
fi

echo "=========================================="
echo "SpMV Benchmark Analysis"
echo "=========================================="
echo ""

# Find unique (n, nz) pairs from filenames
echo "Extracting configurations from filenames..."
configs=$(ls "$results_dir"/seq_n*.txt 2>/dev/null | while read f; do basename "$f" | sed 's/seq_n\([0-9]*\)_nz\([0-9]*\)\.txt/\1 \2/'; done | sort -u)

if [ -z "$configs" ]; then
    echo "Warning: Could not extract configuration from filenames."
    echo "  the results directory contains the following files:"
    ls -1 "$results_dir"/*.txt | head -20
    exit 1
fi

# Process each configuration
echo "$configs" | while read n nz; do
    echo ""
    echo "=============================================================="
    printf "Configuration: n=%s, nz=%s\n" "$n" "$nz"
    echo "=============================================================="
    
    # Find sequential result
    seq_file=$(ls "$results_dir"/seq_n${n}_nz${nz}.txt 2>/dev/null)
    
    if [ -z "$seq_file" ]; then
        echo "  No sequential result found"
        continue
    fi
    
    echo ""
    echo "--- Correctness Verification ---"
    
    seq_checksum=$(grep "checksum=" "$seq_file" | head -1 | cut -d= -f2)
    seq_rayleigh=$(grep "rayleigh=" "$seq_file" | head -1 | cut -d= -f2)
    seq_time=$(grep "Time (sec)" "$seq_file" | awk '{print $NF}')
    
    echo "Sequential:"
    printf "  checksum=%s\n" "$seq_checksum"
    printf "  rayleigh=%s\n" "$seq_rayleigh"
    printf "  time=%.4f sec\n" "$seq_time"
    
    # Find all threaded results for this config
    thread_files=$(ls "$results_dir"/thread_n${n}_nz${nz}_t*.txt 2>/dev/null | sort -V)
    
    if [ -z "$thread_files" ]; then
        echo "  No threaded results found"
        continue
    fi
    
    echo ""
    echo "--- Threaded Results ---"
    printf "%-8s %-12s %-10s %-12s %-10s\n" "Threads" "Time(sec)" "Speedup" "Efficiency" "Checksum"
    echo "------- ----------- --------- ----------- ----------"
    
    for tfile in $thread_files; do
        # Extract thread count from filename: thread_n<n>_nz<nz>_t<t>.txt
        threads=$(echo $(basename "$tfile") | sed 's/.*_t\([0-9]*\)\.txt/\1/')
        
        t_checksum=$(grep "checksum=" "$tfile" | head -1 | cut -d= -f2)
        t_rayleigh=$(grep "rayleigh=" "$tfile" | head -1 | cut -d= -f2)
        t_time=$(grep "Time (sec)" "$tfile" | awk '{print $NF}')
        
        if [ -z "$t_time" ] || [ "$t_time" = "0" ]; then
            printf "%-8s %-12s %-10s %-12s %-10s\n" "$threads" "N/A" "N/A" "N/A" "FAILED"
            continue
        fi
        
        # Calculate speedup and efficiency
        speedup=$(awk "BEGIN {printf \"%.2f\", $seq_time / $t_time}")
        efficiency=$(awk "BEGIN {printf \"%.1f\", ($seq_time / $t_time) / $threads * 100}")
        
        # Check correctness
        if [ "$t_checksum" = "$seq_checksum" ]; then
            checksum_status="✓"
        else
            checksum_status="✗MISMATCH"
        fi
        
        printf "%-8s %-12.4f %-10sx %-11.1f%% %-10s\n" "$threads" "$t_time" "$speedup" "$efficiency" "$checksum_status"
    done
    
    # Compare vector dumps for the sequential and first threaded run
    seq_dump="$results_dir/seq_vec_n${n}_nz${nz}.dump"
    first_tfile=$(echo "$thread_files" | head -1)
    first_threads=$(echo $(basename "$first_tfile") | sed 's/.*_t\([0-9]*\)\.txt/\1/')
    thread_dump="$results_dir/thread_vec_n${n}_nz${nz}_t${first_threads}.dump"
    
    echo ""
    echo "--- Vector Dump Comparison ---"
    compare_dumps "$seq_dump" "$thread_dump"
    
    t1_rayleigh=$(grep "rayleigh=" "$first_tfile" | head -1 | cut -d= -f2)
    
    echo ""
    echo "--- Rayleigh Quotient ---"
    printf "Sequential: %.15e\n" "$seq_rayleigh"
    printf "Threaded(t=%s): %.15e\n" "$first_threads" "$t1_rayleigh"
done

echo ""
echo "=============================================================="
echo "Analysis complete. All results in 'results/' directory."
echo "=============================================================="
echo ""
echo "To view raw output:"
echo "  grep 'Time (sec)' results/*.txt"
echo "  grep 'checksum=' results/*.txt"
echo ""
