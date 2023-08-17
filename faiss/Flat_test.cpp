#include <cstdio>
#include <cstdlib>
#include <deque>
// #include <random>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/IDSelector.h>

// 64-bit int
using idx_t = faiss::idx_t;

int main()
{
    int d = 64;   // dimension
    int nb = 100; // database size
    int nq = 10;  // nb of queries

    // std::mt19937 rng;
    // std::uniform_real_distribution<> distrib;

    // float *xb = new float[d * nb];
    std::vector<float> xb = std::vector<float>(d * nb);
    std::vector<idx_t> xbIndex = std::vector<idx_t>(nb);
    // std::deque<float> xb();
    // std::deque<float> xb = std::deque<float>(d * nb);

    float *xq = new float[d * nq];
    int start = 20, end = 50;
    faiss::IDSelectorRange rangeIndex(start, end);

    for (int i = 0; i < nb; i++)
    {
        for (int j = 0; j < d; j++)
        {
            xb[d * i + j] = i;
        }
        xbIndex[i] = 10 - i / 10;
        // xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++)
    {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = i * 10;
        // xq[d * i] += i / 1000.;
    }
    // faiss::IndexIVFPQ;

    faiss::IndexFlatL2 index(d); // call constructor
    printf("is_trained = %s\n", index.is_trained ? "true" : "false");
    index.add(nb, xb.data()); // add vectors to the index
    index.add(nb, xb.data());
    // index.add_with_ids(nb, xb.data(), xbIndex.data());
    printf("ntotal = %zd\n", index.ntotal);
    index.remove_ids(rangeIndex);
    printf("ntotal = %zd\n", index.ntotal);

    int k = 4;

    { // sanity check: search 5 first vectors of xb
        idx_t *I = new idx_t[k * 5];
        float *D = new float[k * 5];

        index.search(5, xb.data(), k, D, I);

        // print results
        printf("I=\n");
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < k; j++)
                printf("%7g ", D[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    { // search xq
        idx_t *I = new idx_t[k * nq];
        float *D = new float[k * nq];

        index.search(nq, xq, k, D, I);

        // print results
        printf("I (nq first results)=\n");
        for (int i = 0; i < nq; i++)
        {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        // printf("I (5 last results)=\n");
        // for (int i = nq - 5; i < nq; i++)
        // {
        //     for (int j = 0; j < k; j++)
        //         printf("%5zd ", I[i * k + j]);
        //     printf("\n");
        // }

        delete[] I;
        delete[] D;
    }

    // delete[] xb;
    delete[] xq;

    return 0;
}

// int main()
// {
//     int d = 64;      // dimension
//     int nb = 100000; // database size
//     int nq = 10000;  // nb of queries

//     std::mt19937 rng;
//     std::uniform_real_distribution<> distrib;

//     float *xb = new float[d * nb];
//     float *xq = new float[d * nq];

//     for (int i = 0; i < nb; i++)
//     {
//         for (int j = 0; j < d; j++)
//             xb[d * i + j] = distrib(rng);
//         xb[d * i] += i / 1000.;
//     }

//     for (int i = 0; i < nq; i++)
//     {
//         for (int j = 0; j < d; j++)
//             xq[d * i + j] = distrib(rng);
//         xq[d * i] += i / 1000.;
//     }

//     faiss::IndexFlatL2 index(d); // call constructor
//     printf("is_trained = %s\n", index.is_trained ? "true" : "false");
//     index.add(nb, xb); // add vectors to the index
//     printf("ntotal = %zd\n", index.ntotal);

//     int k = 4;

//     { // sanity check: search 5 first vectors of xb
//         idx_t *I = new idx_t[k * 5];
//         float *D = new float[k * 5];

//         index.search(5, xb, k, D, I);

//         // print results
//         printf("I=\n");
//         for (int i = 0; i < 5; i++)
//         {
//             for (int j = 0; j < k; j++)
//                 printf("%5zd ", I[i * k + j]);
//             printf("\n");
//         }

//         printf("D=\n");
//         for (int i = 0; i < 5; i++)
//         {
//             for (int j = 0; j < k; j++)
//                 printf("%7g ", D[i * k + j]);
//             printf("\n");
//         }

//         delete[] I;
//         delete[] D;
//     }

//     { // search xq
//         idx_t *I = new idx_t[k * nq];
//         float *D = new float[k * nq];

//         index.search(nq, xq, k, D, I);

//         // print results
//         printf("I (5 first results)=\n");
//         for (int i = 0; i < 5; i++)
//         {
//             for (int j = 0; j < k; j++)
//                 printf("%5zd ", I[i * k + j]);
//             printf("\n");
//         }

//         printf("I (5 last results)=\n");
//         for (int i = nq - 5; i < nq; i++)
//         {
//             for (int j = 0; j < k; j++)
//                 printf("%5zd ", I[i * k + j]);
//             printf("\n");
//         }

//         delete[] I;
//         delete[] D;
//     }

//     delete[] xb;
//     delete[] xq;

//     return 0;
// }