
#!/bin/bash

# AWK-based Class3D analysis (no RELION utilities required)
# Usage: ./awk_Class3D_count.sh [job_number]

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'; BOLD='\033[1m'

if [ $# -eq 1 ]; then job_num=$1; else
    echo -e "${CYAN}Enter job number:${NC}"; read job_num; fi

if ! [[ "$job_num" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: Job number must be numeric${NC}"; exit 1; fi

job_dir="Class3D/job$(printf "%03d" $((10#$job_num)))"
[ ! -d "$job_dir" ] && echo -e "${RED}Error: Directory $job_dir not found${NC}" && exit 1

echo -e "${CYAN}${BOLD}=== Analyzing Class3D Job ${job_num} ===${NC}"
echo -e "${BLUE}Directory: ${job_dir}${NC}\n"

iterations=(020 023 024 025)

# Get total particles
for iter in "${iterations[@]}"; do
    data_file="${job_dir}/run_it${iter}_data.star"
    if [ -f "$data_file" ]; then
        total_particles=$(awk 'NF>20 && $1 ~ /^[0-9]/ {count++} END {print count}' "$data_file")
        echo -e "${GREEN}Total particles in dataset: ${total_particles}${NC}\n"
        break
    fi
done

# Analyze iterations
for iter in "${iterations[@]}"; do
    model_file="${job_dir}/run_it${iter}_model.star"
    data_file="${job_dir}/run_it${iter}_data.star"

    if [ -f "$model_file" ]; then
        echo -e "${YELLOW}${BOLD}======= Iteration ${iter#0} =======${NC}"
        [ -f "$data_file" ] && iter_particles=$(awk 'NF>20 && $1 ~ /^[0-9]/ {count++} END {print count}' "$data_file") \
            && echo -e "${CYAN}Particles in this iteration: ${iter_particles}${NC}\n"

        echo -e "${BOLD}Class Statistics:${NC}"
        echo "Class | Particles |   (%)  | Ang.Acc | Trans.Acc | Resolution"
        echo "------|-----------|--------|---------|-----------|------------"

        # Parse model.star for class stats from data_model_classes
        awk -v total="$total_particles" '
        BEGIN {inblock=0; classnum=0}
/^data_model_classes/ {inblock=1; next}
/^data_/ && inblock && !/^data_model_classes/ {exit}
inblock && /^loop_/ {next}
inblock && /^_rln/ {next}
inblock && $1 ~ /^Class3D/ {
    classnum++
    dist=$2; rot=$3; trans=$4; res=$5
    particles=int(dist*total); perc=dist*100
    printf "  %2d  | %9d | %6.2f | %7s | %9s | %10s\n", classnum, particles, perc, rot, trans, res
}' "$model_file"

        # Actual particle distribution from data.star
        if [ -f "$data_file" ]; then
            echo -e "\n${CYAN}Actual particle distribution:${NC}"
            awk '
            /^data_particles/ {inblock=1; next}
/^data_/ && inblock && !/^data_particles/ {exit}
inblock && /^loop_/ {next}
inblock && /^_rln/ {header[$1]=++colcount; next}
inblock && $1 ~ /^[0-9]/ {
    class=$(header["_rlnClassNumber"])
    count[class]++; total++
}
END {
    for(c in count) printf "  Class %d: %6d particles (%5.1f%%)\n", c, count[c], 100*count[c]/total
    printf "  Total:   %6d particles\n", total
}' "$data_file"
        fi
        echo ""
    else
        echo -e "${RED}Iteration ${iter#0} not found${NC}\n"
    fi
done

# Resolution progression
echo -e "${YELLOW}${BOLD}=== Resolution Progression Across Iterations ===${NC}\n"
echo "Class | Iter 20 | Iter 23 | Iter 24 | Iter 25 | Best"
echo "------|---------|---------|---------|---------|--------"

declare -A best_res best_iter
for class in 1 2 3 4; do
    printf "  %2d  |" $class
    best_res[$class]=999; best_iter[$class]="--"
    for iter in "${iterations[@]}"; do
        model_file="${job_dir}/run_it${iter}_model.star"
        if [ -f "$model_file" ]; then
            res=$(awk -v c=$class '
            BEGIN {inblock=0; classnum=0}
/^data_model_classes/ {inblock=1; next}
/^data_/ && inblock && !/^data_model_classes/ {exit}
inblock && /^loop_/ {next}
inblock && /^_rln/ {next}
inblock && $1 ~ /^Class3D/ {
    classnum++
    if(classnum==c) print $5
}' "$model_file")
            if [ -n "$res" ]; then
                printf " %7s |" "$res Å"
                (( $(echo "$res < ${best_res[$class]}" | bc -l) )) && best_res[$class]=$res && best_iter[$class]=${iter#0}
            else
                printf "   N/A   |"
            fi
        else
            printf "   ---   |"
        fi
    done
    if [ "${best_res[$class]}" != "999" ]; then
        printf " %s Å (it%s)\n" "${best_res[$class]}" "${best_iter[$class]}"
    else
        printf " ---\n"
    fi
done

# Particle Redistribution Summary
echo -e "\n${YELLOW}${BOLD}=== Particle Redistribution Summary ===${NC}\n"
first_iter=""; last_iter=""
for iter in "${iterations[@]}"; do
    [ -f "${job_dir}/run_it${iter}_data.star" ] && { [ -z "$first_iter" ] && first_iter=$iter; last_iter=$iter; }
done
if [ -n "$first_iter" ] && [ -n "$last_iter" ] && [ "$first_iter" != "$last_iter" ]; then
    echo -e "Comparing iteration ${first_iter#0} → ${last_iter#0}:\n"
    for class in 1 2 3 4; do
        first_count=$(awk -v c=$class '
        /^data_particles/ {inblock=1; next}
/^data_/ && inblock && !/^data_particles/ {exit}
inblock && /^loop_/ {next}
inblock && /^_rln/ {header[$1]=++colcount; next}
inblock && $1 ~ /^[0-9]/ && $(header["_rlnClassNumber"])==c {count++}
END {print count+0}' "${job_dir}/run_it${first_iter}_data.star")

        last_count=$(awk -v c=$class '
        /^data_particles/ {inblock=1; next}
/^data_/ && inblock && !/^data_particles/ {exit}
inblock && /^loop_/ {next}
inblock && /^_rln/ {header[$1]=++colcount; next}
inblock && $1 ~ /^[0-9]/ && $(header["_rlnClassNumber"])==c {count++}
END {print count+0}' "${job_dir}/run_it${last_iter}_data.star")

        if [ "$first_count" -gt 0 ] || [ "$last_count" -gt 0 ]; then
            change=$((last_count - first_count))
            color=$NC; sign=""
            [ $change -gt 0 ] && color=$GREEN && sign="+"
            [ $change -lt 0 ] && color=$RED
            printf "Class %d: %6d → %6d  " $class $first_count $last_count
            echo -e "${color}(${sign}${change} particles)${NC}"
        fi
    done
fi

echo -e "\n${GREEN}Analysis complete!${NC}"
                                               