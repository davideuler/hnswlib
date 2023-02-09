#include "../../hnswlib/hnswlib.h"
#include <thread>

#include <chrono>
#include <ctime>
#include <string>
#include <iostream>

#ifdef _WIN32
#include <io.h>
   #define access    _access_s
#else
#include <unistd.h>
#endif

#include <sys/stat.h>

using namespace std::literals::string_literals;

int exist(const char *name)
{
    struct stat   buffer;
    return (stat (name, &buffer) == 0);
}

template <
        class result_t   = std::chrono::milliseconds,
        class clock_t    = std::chrono::steady_clock,
        class duration_t = std::chrono::milliseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


// Filter that allows labels divisible by divisor
class PickDivisibleIds: public hnswlib::BaseFilterFunctor {
unsigned int divisor = 1;
 public:
    PickDivisibleIds(unsigned int divisor): divisor(divisor) {
        assert(divisor != 0);
    }
    bool operator()(hnswlib::labeltype label_id) {
        return label_id % divisor == 0;
    }
};


int main() {
    int dim = 128;               // Dimension of the elements
    int max_elements = 500000;   // Maximum number of elements, should be known beforehand
    int M = 32;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 500;  // Controls index search speed/build speed tradeoff
    int num_threads = 20;       // Number of threads for operations with index
    hnswlib::L2Space space(dim);

    std::string hnsw_path =  "hnsw_mt_";
    hnsw_path += std::to_string(M) + "_"s + std::to_string(ef_construction) + "_"s + std::to_string(max_elements)  + ".bin"s;

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    if(!exist(hnsw_path.c_str())){
        // Initing index
        hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

        auto timenow = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << "starting build index, total elements:" << max_elements << " at " << ctime(&timenow) << std::endl;
        auto start = std::chrono::steady_clock::now();

        // Add data to index
        ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
            alg_hnsw->addPoint((void*)(data + dim * row), row);
        });

        std::cout << "Index duration elapsed(ms)=" << since(start).count() << " docs:" << max_elements  << std::endl;

        // Serialize index
        alg_hnsw->saveIndex(hnsw_path);
        delete alg_hnsw;
    }

    // Deserialize index and check recall
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    alg_hnsw->setEf(512);

    // Create filter that allows only even labels
    PickDivisibleIds pickIdsDivisibleByTwo(2);

    auto search_start = std::chrono::steady_clock::now();
    // Query the elements for themselves with filter and check returned labels
    int k = 10;
    int query_count = 100;
    std::vector<hnswlib::labeltype> neighbors(query_count * k);
    std::vector<float> scores(query_count * k);
    ParallelFor(0, query_count, num_threads, [&](size_t row, size_t threadId) {
//        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + dim * row, k, &pickIdsDivisibleByTwo);
        std::vector<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnnCloserFirst(data + row * dim, k, &pickIdsDivisibleByTwo);

        for (int i = 0; i < k; i++) {
            auto index = row * k + i;
            scores[index] = result.at(i).first;
            neighbors[index] = result.at(i).second;
        }
    });
    std::cout << "Search duration elapsed(ms)=" << since(search_start).count() << " queries:" << query_count << std::endl;

    for (int row = 0; row < query_count; row++) {
        for (int i = 0; i < k; i++) {
            auto index = row * k + i;
            hnswlib::labeltype label = neighbors[index];
            if (label % 2 == 1) {
                std::cout << "Error: found odd label\n";
            }
            std::cout << label << ":" << scores[index] << ",";
        }
        std::cout << std::endl << "====" << std::endl;
        if (row % 2 == 0) { // for odd id, and suitable ef value, it should be true:
            if (row != neighbors[row * k]) {
                std::cout << "incorrect for row:" << row << std::endl;
            }
            assert(row == neighbors[row * k]); // test the result
        }
    }


    delete[] data;
    delete alg_hnsw;
    return 0;
}
