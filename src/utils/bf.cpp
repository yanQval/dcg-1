#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>
using namespace std;

const int maxN = 50;
const int maxM = 50;
const int MAX_BATCH_SIZE = 35;


class BruteForce_Solver{

    int n, m, k;
    double *f, *g, v_max;
    int edges_from[maxN], edges_to[maxN], actions[maxN];

    void dfs(int t, double *best_actions){
        if(t == n){
            double v = 0;
            for(int i = 0; i < n; i++)
                v += f[i * m + actions[i]];
            for(int i = 0; i < k; i++)
                v += g[i * m * m + actions[edges_from[i]] * m + actions[edges_to[i]]] * 1;
            if (v > v_max){
                v_max = v;
                for(int i = 0; i < n; i++)
                    best_actions[i] = actions[i];
            }
            return;
        }
        for (int i = 0 ; i < m; i++){
            actions[t] = i;
            dfs(t + 1, best_actions);
        }
    }
    public:
    void solve(double *py_f, double *py_g, double *py_edges_from, double *py_edges_to, double *best_actions, int py_n, int py_m, int py_k){
        n = py_n, m = py_m, k = py_k;
        f = py_f, g = py_g;
        for (int i = 0 ; i < k; i++){
            edges_from[i] = int(py_edges_from[i]);
            edges_to[i] = int(py_edges_to[i]);
        }
        v_max = -1e30;
        dfs(0, best_actions);
    }
};

BruteForce_Solver bfsolver[MAX_BATCH_SIZE];

extern "C" void
solve(double *py_f, double *py_g, double *py_edges_from, double *py_edges_to, double *best_actions, int bs, int py_n, int py_m, int py_k){
#pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_BATCH_SIZE)
    for (int i = 0; i < bs; i++)
        bfsolver[i].solve(py_f + i * py_n * py_m, py_g + i * py_k * py_m * py_m, py_edges_from, py_edges_to, best_actions + py_n, py_n, py_m, py_k);
}