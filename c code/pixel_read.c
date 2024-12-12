#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void print_individual(int * pixels, size_t size) {
    for (int i = 0; i < size*3; i+=3) {
        printf("[");
        for (int k = 0; k < 3; k++) printf("%d ", pixels[i+k]);
        printf("],\n");
    }
}

float evaluate_fitness(int * ind_pixels, int * orig_pixels, double goal, size_t size) {
    float pixels_similar = 0;
    for (int i = 0; i < size*3; i+=3) {
        bool similar = true;
        for (int p = 0; p < 3 && similar; p++) {
            if (ind_pixels[i+p] != orig_pixels[i+p]) similar = false;
        }
        if (similar) pixels_similar++;
    }
    float percent_similar = (pixels_similar / size) * 100;
    float fitness = fabs(percent_similar - goal);
    return fitness;
}

// random int generator used because c rand() does not generate large numbers
int xorshift32(int seed) {
    int x = seed;
    x ^= (x << 13);
    x ^= (x >> 17);
    x ^= (x << 5);
    return abs(x);
}

int* mass_swap(int * pixels, int seed, size_t size) {
    int ** swapped_pixels = (int**) malloc(size * sizeof(int*));
    for (int i = 0, p = 0; i < size*3; i+=3, p++) {
        swapped_pixels[p] = (int*) malloc(3 * sizeof(int));
        for (int k = 0; k < 3; k++) swapped_pixels[p][k] = pixels[i+k];
    }
    
    int num_swaps = xorshift32(seed) % size;
    int reuse_seed = xorshift32(num_swaps) % size;

    for (int i = 0; i < num_swaps; i++) {
        reuse_seed = xorshift32(reuse_seed | rand());
        int pos1 = reuse_seed % size;
        reuse_seed = xorshift32(reuse_seed | rand());
        int pos2 = reuse_seed % size;

        int * tmp = swapped_pixels[pos1];
        swapped_pixels[pos1] = swapped_pixels[pos2];
        swapped_pixels[pos2] = tmp;
    }

    int * flattened = (int*) malloc(size*3*sizeof(int));
    for (int i = 0, j = 0; j < size; j++) {
        for (int p = 0; p < 3; i++, p++)
            flattened[i] = swapped_pixels[j][p];
        free(swapped_pixels[j]);
    }
    free(swapped_pixels);

    return flattened;
}

int* smart_swap(int * pixels, int seed, int max_swap, size_t size) {
    int ** swapped_pixels = (int**) malloc(size * sizeof(int*));
    for (int i = 0, p = 0; i < size*3; i+=3, p++) {
        swapped_pixels[p] = (int*) malloc(3 * sizeof(int));
        for (int k = 0; k < 3; k++) swapped_pixels[p][k] = pixels[i+k];
    }
    
    int num_swaps = xorshift32(seed) % max_swap;
    int reuse_seed = xorshift32(num_swaps) % size;

    for (int i = 0; i < num_swaps; i++) {
        reuse_seed = xorshift32(reuse_seed | rand());
        int pos1 = reuse_seed % size;
        reuse_seed = xorshift32(reuse_seed | rand());
        int pos2 = reuse_seed % size;

        int * tmp = swapped_pixels[pos1];
        swapped_pixels[pos1] = swapped_pixels[pos2];
        swapped_pixels[pos2] = tmp;
    }

    int * flattened = (int*) malloc(size*3*sizeof(int));
    for (int i = 0, j = 0; j < size; j++) {
        for (int p = 0; p < 3; i++, p++)
            flattened[i] = swapped_pixels[j][p];
        free(swapped_pixels[j]);
    }
    free(swapped_pixels);

    return flattened;
}

// returns position of tournament winner
int tournament_select(float * fitness, int opps, size_t size) {
    int winner = rand() % size;
    for (int k = 0; k < opps-1; k++) {
        int opp = rand() % size;
        if (fitness[winner] < fitness[opp])
            winner = rand() % 11 <= 8 ? winner : opp;
        else if (fitness[winner] == fitness[opp])
            winner = rand() % 2 == 0 ? winner : opp;
        else
            winner = rand() % 11 <= 2 ? winner : opp;
    } 
    return winner;
}

// used in crossovers and greedy
int find_pixel_ptr(int ** parent, int * pixel, int start, int end) {
    for (int i = start; i < end; i++) {
        if (parent[i] == pixel) return i;
    }
    return -1;
}

int* pmx_cross(int * parent1, int * parent2, size_t size) {
    int ** p_1 = (int**) malloc(size * sizeof(int*));
    int ** p_2 = (int**) malloc(size * sizeof(int*));
    int ** child = (int**) malloc(size * sizeof(int*));

    // helps keep track of which pixels have been copied
    bool copied[size];

    for (int i = 0; i < size; i++) {
        p_1[i] = (int*) malloc(3 * sizeof(int));
        p_2[i] = (int*) malloc(3 * sizeof(int));
        child[i] = (int*) malloc(3 * sizeof(int));
        for (int p = 0; p < 3; p++) {
            p_1[i][p] = parent1[i*3+p];
            p_2[i][p] = parent2[i*3+p];
        }
    }
    int pos1 = xorshift32(rand()) % size;
    int pos2 = xorshift32(rand()) % size;
    while (pos1 == pos2) pos2 = xorshift32(rand()) % size;
    if (pos1 > pos2) {
        int tmp = pos1;
        pos1 = pos2;
        pos2 = pos1;
    }
    // copy random segment from P1 to child
    for (int i = pos1; i <= pos2; i++) {
        child[i] = p_1[i];
        copied[i] = true;
    }
    // may have issue with duplicates. if so in find_pixel_ptr compare address of pixel
    for (int i = pos1; i <= pos2; i++) {
        int child_pos = find_pixel_ptr(child, p_2[i], 0, size);
        if (child_pos != -1) {
            child_pos = find_pixel_ptr(p_2, p_1[i], 0, size);
            while (pos1 <= child_pos && child_pos <= pos2) 
                child_pos = find_pixel_ptr(p_2, p_1[child_pos], 0, size);
            child[child_pos] = p_1[i];
            copied[i] = true;
        }
    }

    for (int i = 0; i < size; i++) {
        if (!copied[i]) child[i] = p_2[i];
    }

    int * flattened = (int*) malloc(size*3*sizeof(int));
    for (int i = 0, j = 0; j < size; j++) {
        for (int p = 0; p < 3; i++, p++) flattened[i] = child[j][p];
        free(child[j]);
    }
    
    free(child);
    return flattened;
}

// pmx cross has range limit
int* smart_pmx_cross(int * parent1, int * parent2, size_t size, size_t max_cross) {
    int ** p_1 = (int**) malloc(size * sizeof(int*));
    int ** p_2 = (int**) malloc(size * sizeof(int*));
    int ** child = (int**) malloc(size * sizeof(int*));

    // helps keep track of which pixels have been copied
    bool copied[size];

    for (int i = 0; i < size; i++) {
        p_1[i] = (int*) malloc(3 * sizeof(int));
        p_2[i] = (int*) malloc(3 * sizeof(int));
        child[i] = (int*) malloc(3 * sizeof(int));
        for (int p = 0; p < 3; p++) {
            p_1[i][p] = parent1[i*3+p];
            p_2[i][p] = parent2[i*3+p];
        }
    }
    int pos1 = xorshift32(rand()) % size;
    int pos2 = xorshift32(rand()) % size;

    while(abs(pos1-pos2) >= max_cross || pos1 == pos2) pos2 = xorshift32(rand()) % size;

    if (pos1 > pos2) {
        int tmp = pos1;
        pos1 = pos2;
        pos2 = pos1;
    }
    // copy random segment from P1 to child
    for (int i = pos1; i <= pos2; i++) {
        child[i] = p_1[i];
        copied[i] = true;
    }
    for (int i = pos1; i <= pos2; i++) {
        int child_pos = find_pixel_ptr(child, p_2[i], 0, size);
        if (child_pos != -1) {
            child_pos = find_pixel_ptr(p_2, p_1[i], 0, size);
            while (pos1 <= child_pos && child_pos <= pos2) 
                child_pos = find_pixel_ptr(p_2, p_1[child_pos], 0, size);
            child[child_pos] = p_1[i];
            copied[i] = true;
        }
    }

    for (int i = 0; i < size; i++) {
        if (!copied[i]) child[i] = p_2[i];
    }

    int * flattened = (int*) malloc(size*3*sizeof(int));
    for (int i = 0, j = 0; j < size; j++) {
        for (int p = 0; p < 3; i++, p++) flattened[i] = child[j][p];
        free(child[j]);
    }
    
    free(child);
    return flattened;
}

int* order_cross(int * parent1, int * parent2, size_t size) {
    int ** p_1 = (int**) malloc(size * sizeof(int*));
    int ** p_2 = (int**) malloc(size * sizeof(int*));
    int ** child = (int**) malloc(size * sizeof(int*));

    // helps keep track of which pixels have been copied
    bool copied[size];

    for (int i = 0; i < size; i++) {
        p_1[i] = (int*) malloc(3 * sizeof(int));
        p_2[i] = (int*) malloc(3 * sizeof(int));
        child[i] = (int*) malloc(3 * sizeof(int));
        for (int p = 0; p < 3; p++) {
            p_1[i][p] = parent1[i*3+p];
            p_2[i][p] = parent2[i*3+p];
        }
    }
    int pos1 = xorshift32(rand()) % size;
    int pos2 = xorshift32(rand()) % size;

    while(pos1 == pos2) pos2 = xorshift32(rand()) % size;
    
    if (pos1 > pos2) {
        int tmp = pos1;
        pos1 = pos2;
        pos2 = pos1;
    }

    // copy random segment from P1 to child
    for (int i = pos1; i <= pos2; i++) child[i] = p_1[i];

    // copy rest from second parent in order
    int p2_iter = pos2+1;
    int child_iter = p2_iter;

    for (; p2_iter < size && child_iter < size; p2_iter++) {
        if (find_pixel_ptr(child, p_2[p2_iter], pos1, pos2) == -1) {
            child[child_iter] = p_2[p2_iter];
            child_iter++;
        }
    }
    p2_iter = 0;
    for (; child_iter < size; p2_iter++) {
        if (find_pixel_ptr(child, p_2[p2_iter], pos1, pos2) == -1) {
            child[child_iter] = p_2[p2_iter];
            child_iter++;
        }
    }
    child_iter = 0;
    for (; child_iter < pos1; p2_iter++) {
        if (find_pixel_ptr(child, p_2[p2_iter], pos1, pos2) == -1) {
            child[child_iter] = p_2[p2_iter];
            child_iter++;
        }
    }

    int * flattened = (int*) malloc(size*3*sizeof(int));
    for (int i = 0, j = 0; j < size; j++) {
        for (int p = 0; p < 3; i++, p++) flattened[i] = child[j][p];
        free(child[j]);
    }
    
    free(child);
    return flattened;
}

// should be better but has similar result. also much much slower
// however may prevent pixel loss
/* int* greedy_generate(int * original, double goal, int seed, size_t size) {
    srand(time(NULL));
    int ** orig = (int**) malloc(size * sizeof(int*));
    int ** gsol = (int**) malloc(size * sizeof(int*));
    int ** unused = (int**) malloc(size * sizeof(int*));
    bool * placed = (bool*) malloc(size * sizeof(bool));

    int unused_len = size;
    
    for (int i = 0; i < size; i++) {
        orig[i] = (int*) malloc(3 * sizeof(int));
        gsol[i] = (int*) malloc(3 * sizeof(int));
        unused[i] = (int*) malloc(3 * sizeof(int));
        for (int p = 0; p < 3; p++) orig[i][p] = original[i*3+p];
        unused[i] = orig[i];
        placed[i] = false;
    }

    // go through image to place orig pixels
    for (int i = 0; i < size; i++) {
        if (rand() % 101 <= goal) {
            gsol[i] = orig[i];
            placed[i] = true;
            int pos = find_pixel_ptr(unused, orig[i], 0, size);
            int * tmp = unused[pos];
            unused[pos] = unused[--unused_len];
            unused[unused_len] = tmp;
        }
    }

    // go through image to fill with random pixels
    int reuse_seed = xorshift32(seed) % size;
    for (int i = 0; i < size; i++) {
        if (!placed[i]) {
            int pos = xorshift32(reuse_seed | rand()) % unused_len;
            gsol[i] = unused[pos];
            reuse_seed = xorshift32(reuse_seed | rand());
            int * tmp = unused[pos];
            unused[pos] = unused[--unused_len];
            unused[unused_len] = tmp;
        }
    }
    int * flattened = (int*) malloc(size*3*sizeof(int));
    for (int i = 0, j = 0; j < size; j++) {
        for (int p = 0; p < 3; i++, p++) flattened[i] = gsol[j][p];
        free(orig[j]);
    }
    free(orig);
    free(gsol);
    free(unused);
    return flattened;
} */

// working version
int* greedy_generate(int * original, double goal, int seed, size_t size) {
    srand(time(NULL));
    int ** gsol = (int**) malloc(size * sizeof(int*));
    int * match = (int*) malloc(size * sizeof(int));
    int * unused = (int*) malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        gsol[i] = (int*) malloc(3 * sizeof(int));
        unused[i] = i;
        for (int p = 0; p < 3; p++) gsol[i][p] = original[i*3+p];
    }
    int reuse_seed = xorshift32(seed) % size;
    int match_len  = 0;
    int unused_len = size;
    for (int i = 0; i < size; i++) {
        int pos = i;
        if (rand() % 101 <= goal) match[match_len++] = i;
        else {
            pos = xorshift32(reuse_seed | rand()) % unused_len;
            gsol[i][0] = gsol[unused[pos]][0];
            gsol[i][1] = gsol[unused[pos]][1];
            gsol[i][2] = gsol[unused[pos]][2];
            reuse_seed = xorshift32(reuse_seed | rand());
        }
        int tmp = unused[pos];
        unused[pos] = unused[--unused_len];
        unused[unused_len] = tmp;
    }
    int * flattened = (int*) malloc(size*3*sizeof(int));
    for (int i = 0, j = 0; j < size; j++) {
        for (int p = 0; p < 3; i++, p++) flattened[i] = gsol[j][p];
        free(gsol[j]);
    }
    free(gsol);
    free(match);
    free(unused);
    return flattened;
}