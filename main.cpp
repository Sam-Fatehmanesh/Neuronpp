#include "Neuron.hpp"
#include <string>
#include <utility>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>
#include <fstream>

//implement return regular NN
//combine darwin + calc

using namespace af;
using namespace std;

class CSVReader
{
    std::string fileName;
    std::string delimeter;
public:
    explicit CSVReader(string filename, string delm = ",")
    {
        fileName = move(filename);
        delimeter = move(delm);
    }
    // Function to fetch data from a CSV File
    vector<vector<float>> getData()
    {
        ifstream data(fileName);
        vector<vector<float> > dataList;
        string line;
        while (std::getline(data, line))
        {
            vector<string> vec;
            boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
            vector<float> v;
            v.reserve(vec.size());
            for (auto & i : vec)
            {
                v.push_back(std::stof(i));
            }
            dataList.push_back(v);
        }
        data.close();

        return dataList;
    }
};

int main(){
    vector<vector<float>> in;
    vector<vector<float>> out;

    in.push_back(vector<float>{1,1});
    in.push_back(vector<float>{0,1});
    in.push_back(vector<float>{1,0});
    in.push_back(vector<float>{0,0});
    out.push_back(vector<float>{1});
    out.push_back(vector<float>{0});
    out.push_back(vector<float>{0});
    out.push_back(vector<float>{1});


    auto neuralstructure = NetSpecs(2,1);
    neuralstructure.addlayer(2,"sig");
    neuralstructure.addlayer(1,"sig");

    Darwin species = Darwin(neuralstructure, 200,10);
    species.installdata(in,out,in,out);
    species.evolve(2,0.1,100, true,false);

    cout << "accuracy: " << species.accuracy() << endl;
}