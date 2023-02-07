#include "../../hnswlib/hnswlib.h"
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
int exist(const char *name)
{
  struct stat   buffer;
  return (stat (name, &buffer) == 0);
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


template <
        class result_t   = std::chrono::milliseconds,
        class clock_t    = std::chrono::steady_clock,
        class duration_t = std::chrono::milliseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}


int main() {

    int dim = 128;              // Dimension of the elements
            
    int max_elements = 10000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    std::string hnsw_path = "hnsw.bin";
    
    hnswlib::L2Space space(dim);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47); // make reproducible random numbers
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
            data[i] = distrib_real(rng);
    }


    if(!exist("hnsw.bin")){

        // Initing index
        hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

        auto timenow = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << "starting build index, total elements:" << max_elements << " at " << ctime(&timenow) << std::endl;

        auto start = std::chrono::steady_clock::now();
        
        // Add data to index
        for (int i = 0; i < max_elements; i++) {
            alg_hnsw->addPoint(data + i * dim, i);
        }

        std::cout << "Index duration elapsed(ms)=" << since(start).count() << " docs:" << max_elements  << std::endl;

        // Serialize index
        alg_hnsw->saveIndex(hnsw_path);
        delete alg_hnsw;
    }


    // Deserialize index and check recall
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);

    // Create filter that allows only even labels
    PickDivisibleIds pickIdsDivisibleByTwo(2);

    // Query the elements for themselves with filter and check returned labels
    int k = 100;
    auto search_start = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; i++) {
        std::vector<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnnCloserFirst(data + i * dim, k, &pickIdsDivisibleByTwo);
        for (auto item: result) {
            if (item.second % 2 == 1) std::cout << "Error: found odd label\n";
            std::cout << item.second << ",";
        }
        std::cout << std::endl;
    }
    std::cout << "Search duration elapsed(ms)=" << since(search_start).count() << std::endl;

    delete[] data;
    delete alg_hnsw;
    return 0;
}
