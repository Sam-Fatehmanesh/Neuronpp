#include <arrayfire.h>
#include <memory>
#include <utility>
#include <vector>
#include <iostream>
#include <functional>
#include <memory>
#include <random>
#include <algorithm>
#include <chrono>
#include "Netspecs.hpp"

using namespace af;
using namespace std;

class NeuralTensor
{
private:
    static int randint()
    {
        std::random_device r;
        std::uniform_int_distribution<int> dist;
        int x = dist(r);
        return x;
    }
public:
    vector<af::array> tensor;
    int layernum;
    //netstructure is the structure and composition of the neural network
    explicit NeuralTensor(const NetSpecs& netstructure,af::randomEngine rengine,int initPopN = 1,float initrange = 1)
    {
        //Here each layer is initialized thus initializing the neural tensor
        af::dim4 dim = af::dim4(netstructure.max.first,netstructure.max.second+1,1,initPopN);
        for (int i = 0; i < netstructure.laynum;i++)
        {
            //the dimentions are set for layers of the NN
            dim = af::dim4(netstructure.layers[i].Ncount,netstructure.layers[i].incount+1,1,initPopN);
            //the neural tensorvalues are randomized
            rengine.setSeed(randint());
            tensor.push_back(af::randu(dim, f32, rengine));
        }
        add(-0.5);
        multiply(initrange);
        layernum = tensor.size();
    }
    NeuralTensor(const af::array& inputNN,int numlayer)
    {
        //Here each layer is initialized thus initializing the neural tensor
        for (int i = 0; i < numlayer; i++)
        {
            tensor[i] = inputNN(span,span,i,span);
        }
        layernum = tensor.size();
    }
    NeuralTensor(vector<af::array> inputNN)
    {
        tensor = inputNN;
        layernum = tensor.size();
    }

    void setnetequal(float var,int netnum = 0)
    {
        for (auto & i : tensor)
        {
            i(span,span,0,netnum) = var;
        }
    }
    void setnetequal(NeuralTensor * var,int netnum = 0)
    {
        for (int i = 0; i < tensor.size(); i++)
        {
            tensor[i](span,span,0,netnum) = var->tensor[i](span,span,0,netnum);
        }
    }

    vector<af::array> extractNN(int netnum = 0)
    {
        //creates a new output array
        vector<af::array> output;
        //copies from the tensor
        for (auto & i : tensor)
        {
            output.push_back(i(span, span, 0, netnum));
        }
        return output;
    }
    vector<af::array> extractNN(const af::seq& list)
    {
        //creates a new output array
        vector<af::array> output;
        //copies from the tensor
        for (auto & i : tensor)
        {
            output.push_back(i(span, span, 0, list));
        }
        return output;
    }
    vector<af::array> extractNN(const af::array& list)
    {
        //creates a new output array
        vector<af::array> output;
        //copies from the tensor
        for (auto & i : tensor)
        {
            output.push_back(i(span, span, 0, list));
        }
        return output;
    }

    void add(NeuralTensor * addend)
    {
        for (int i = 0; i < tensor.size();i++)
        {
            tensor[i] += addend->tensor[i];
        }
    }
    void add(float addend)
    {
        for (auto & i : tensor)
        {
            i += addend;
        }
    }

    void multiply(float factor)
    {
        for (auto & i : tensor)
        {
            i = i*factor;
        }
    }
};