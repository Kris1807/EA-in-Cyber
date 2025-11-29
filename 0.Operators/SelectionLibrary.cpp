#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <pybind11/pybind11.h> 
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;


template <typename T>
T select_binary(const vector<T>& population, const vector<double>& fitnesses, mt19937& rng) {
	uniform_int_distribution<int> dist(0, population.size() - 1);
	int i = dist(rng);
	int j = dist(rng);

	if (fitnesses[i] < fitnesses[j]) 
		return population[i];
	else
		return population[j];
}

template <typename T>
T select_roulette(const vector<T>& population, const vector<double>& fitnesses, mt19937& rng) {
	double total_fitness = accumulate(fitnesses.begin(), fitnesses.end(), 0.0);
	if (total_fitness == 0.0) {
		uniform_int_distribution<int> dist(0, population.size() - 1);
		return population[dist(rng)];
	}
	uniform_real_distribution<double> dist(0.0, total_fitness);
	double r = dist(rng);
	double cumulative = 0.0;
	for (int i = 0; i < population.size(); i++) {
		cumulative += fitnesses[i];
		if (cumulative >= r) {
			return population[i];
		}
	}
	return population.back(); 
}

 int tournament_selection(const vector<double>& fitnesses, int tournSize, mt19937& rng) {
	int n = fitnesses.size();
	vector<int> idxs(n);
	for (int i = 0; i < n; i++) {
		idxs[i] = i;
	}
	shuffle(idxs.begin(), idxs.end(), rng);
	idxs.resize(tournSize);
	int best = idxs[0];
	double bestFitness = fitnesses[best];
	for (int i = 1; i < tournSize; i++) {
		int idx = idxs[i];
		if (fitnesses[idx] < bestFitness) {
			best = idx;
			bestFitness = fitnesses[idx];
		}
	}
	return best;
}
 struct RNGWrapper {
     std::mt19937 engine;

     RNGWrapper(unsigned int seed) : engine(seed) {}

     void seed(unsigned int s) { engine.seed(s); }

     int randint(int low, int high) {
         std::uniform_int_distribution<int> dist(low, high);
         return dist(engine);
     }

     double randreal(double low = 0.0, double high = 1.0) {
         std::uniform_real_distribution<double> dist(low, high);
         return dist(engine);
     }
 };

 py::object select_binary_py(py::list population, const vector<double>& fitnesses, mt19937& rng)
 {
     if (py::len(population) == 0)
         throw std::runtime_error("Population cannot be empty");

     py::object first = population[0];

     // INT population
     if (py::isinstance<py::int_>(first)) {
         vector<int> pop_cpp;
         pop_cpp.reserve(py::len(population));
         for (auto item : population)
             pop_cpp.push_back(item.cast<int>());
         int result = select_binary<int>(pop_cpp, fitnesses, rng);
         return py::int_(result);
     }

     // FLOAT population
     if (py::isinstance<py::float_>(first)) {
         vector<double> pop_cpp;
         pop_cpp.reserve(py::len(population));
         for (auto item : population)
             pop_cpp.push_back(item.cast<double>());
         double result = select_binary<double>(pop_cpp, fitnesses, rng);
         return py::float_(result);
     }

     throw std::runtime_error("Population must contain ints or floats");
 }

 py::object select_roulette_py(py::list population, const vector<double>& fitnesses, mt19937& rng)
 {
     if (py::len(population) == 0)
         throw std::runtime_error("Population cannot be empty");

     py::object first = population[0];

     if (py::isinstance<py::int_>(first)) {
         vector<int> pop_cpp;
         for (auto item : population)
             pop_cpp.push_back(item.cast<int>());
         int result = select_roulette<int>(pop_cpp, fitnesses, rng);
         return py::int_(result);
     }

     if (py::isinstance<py::float_>(first)) {
         vector<double> pop_cpp;
         for (auto item : population)
             pop_cpp.push_back(item.cast<double>());
         double result = select_roulette<double>(pop_cpp, fitnesses, rng);
         return py::float_(result);
     }

     throw std::runtime_error("Population must contain ints or floats");
 }


 // --------------------------------------------------------
 //  Python Module Definition
 // --------------------------------------------------------

 PYBIND11_MODULE(selection_functions, m)
 {
     m.doc() = "Genetic Algorithm Selection Functions in C++";
     py::class_<RNGWrapper>(m, "RNG")
         .def(py::init<unsigned int>())
         .def("seed", &RNGWrapper::seed)
         .def("randint", &RNGWrapper::randint)
         .def("randreal", &RNGWrapper::randreal);
     // Unified Python APIs
     m.def("select_binary", &select_binary_py, "Binary tournament selection");
     m.def("select_roulette", &select_roulette_py, "Roulette wheel selection");
     m.def("tournament_selection", &tournament_selection, "Tournament selection (returns index)");

 }
