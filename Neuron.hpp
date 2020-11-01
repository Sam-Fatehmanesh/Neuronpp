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
#include "NeuralTensor.hpp"

using namespace af;
using namespace std;

class Darwin
{
private:
    //initializing
    int popN;//number of NNs
    //random engine
    af::randomEngine rengine = af::randomEngine(af::randomEngineType::AF_RANDOM_ENGINE_PHILOX,randint());
    //main tensor with all NNs
    NeuralTensor * tensor;
    //the error or errors of the NN/NNs
    af::array error;
    //the finished or most optimal NN
    NeuralTensor * optimal{};
    NetSpecs netstructure = NetSpecs(0);
    vector<af::array> trainIn;
    vector<af::array> trainOut;

    void convformat()
    {

    }
    static int randint()
    {
        std::random_device r;
        std::uniform_int_distribution<int> dist;
        int x = dist(r);
        return x;
    }
    static vector<af::array> vectogpu(vector<vector<float>> in)
    {
        vector<af::array> x;
        x.reserve(in.size());
        for (auto & i : in)
        {
            x.emplace_back(i.size(),i.data());
        }
        return x;
    }
    static af::array vectogpuary(vector<vector<float>> in)
    {
        vector<af::array> x;
        x.reserve(in.size());
        for (auto & i : in)
        {
            x.emplace_back(i.size(),i.data());
        }
        af::array y(in.size(),x[0].dims(0));
        for (int i = 0; i < x.size();i++)
        {
            y(i,span) = x[i];
        }
        return y;
    }
    static af::array act(const af::array& x, const string& func)
    {
        af::array y;
        if (func == "sig")
        {
            return sigmoid(x);
        }
        else if(func == "relu") {
            return af::max(0,x);
        } else {
            return x;
        }
    }

public:
    vector<af::array> testIn;
    vector<af::array> testOut;

    explicit Darwin(const NetSpecs& NetStructure,int initPopN = 4,float initrange = 1)
    {
        //netstructure is the structure and composition of the neural network
        netstructure = NetStructure;
        //the population size is first set to the smallest size it can be when inicialized
        popN = initPopN;
        //Here each layer is initialized thus initializing the neural tensor
        tensor = new NeuralTensor(netstructure,rengine,popN,initrange);
        error = af::array(1,1,1,popN);
    }
    ~Darwin() = default;

    vector<af::array> Export(int num = 0, bool print = true)
    {
        if (print) {
            for (auto & i : tensor->tensor)
            {
                af_print(i(span, span, 0, num));
            }
        }
        //creates a new output array
        vector<af::array> output;
        //copies from the tensor
        for (auto & i : tensor->tensor)
        {
            output.push_back(i(span, span, 0, num));
        }
        return output;
    }
    void import(const vector<vector<vector<float>>>& in)
    {
        vector<af::array> newnet;
        newnet.reserve(in.size());
        for (auto & i : in)
        {
            newnet.push_back(vectogpuary(i));
        }
        int i;
        for (int u = netstructure.laynum; u > 0; u--)
        {
            i = netstructure.laynum - u - 1;
            tensor->tensor[i](span, span, 0, span) = newnet[i];
        }

    }

    void traindarwin(float maxmute, float topFrac,int generations,bool debug = false, bool batched = false, int batchsize = 1)
    {
        evolve(maxmute,topFrac,generations,debug,batched,batchsize);
    }
    void trainnewton()
    {

    }

    float accuracy(bool classification = true)
    {
        af::array out;
        af::array errorsmall;
        af::array temp;
        float a = 0;
        if (classification)
        {
            for (int i = 0; i < testIn.size(); i++)
            {
                out = compute(testIn[i],false);
                errorsmall = out - testOut[i];
                errorsmall = af::abs(errorsmall);
                temp = af::sum(errorsmall);
                a += 1-((temp.scalar<float>())/testOut[i].dims(0));
            }
            a /= testIn.size();
        }
        return a;
    }
    void newgenesis(vector<af::array> x,float maxmute,float topfrac)
    {
        for (auto & i : tensor->tensor)
        {
            i = tile(i, 1, 1, 1, int(1 / topfrac));
        }
        for (auto & i : x)
        {
            i = tile(i, 1, 1, 1, popN);
        }
        mutate(maxmute);
    }
    void installdata(vector<vector<float>> trainin, vector<vector<float>> trainout,vector<vector<float>> testin, vector<vector<float>> testout)
    {
        trainIn = vectogpu(move(trainin));
        trainOut = vectogpu(move(trainout));
        testIn = vectogpu(move(testin));
        testOut = vectogpu(move(testout));
    }
    void installtestdata(vector<vector<float>> testin, vector<vector<float>> testout)
    {
        testIn = vectogpu(move(testin));
        testOut = vectogpu(move(testout));
    }
    void optimalerror()
    {
        af_print(error(0));
    }
    af::array compute(const af::array& data,bool all = true)
    {
        if(all)
        {
            af::array intraout;
            af::array base;
            intraout = tile(data,1,1,1,popN);
            af::seq x;
            int modifier;
            for (int u = 0; u < netstructure.laynum; u++)
            {
                modifier = int(tensor->tensor[u].dims(1) - intraout.dims(0));
                if (modifier > 0)
                {
                    base = af::array(modifier,1,1,popN);
                    base(modifier-1,0,0,span) = 1;
                    intraout = af::join(0,intraout,base);
                }
                //cout << tensor->tensor[u].dims() << endl;
                //cout << intraout.dims() << endl;
                intraout = matmul(tensor->tensor[u], intraout);
                intraout = act(intraout,netstructure.layers[u].func);
                /*
                if(netstructure.layers[u].Ncount < netstructure.max.first)
                {
                    intraout(seq(netstructure.layers[u].Ncount,netstructure.max.first-1),span,span,span) = 0;
                }*/

            }
            return intraout(seq(netstructure.outs),span,span,span);
        } else {
            af::array intraout;
            intraout = data;
            af::array base;
            //intraout = tile(intraout,1,1,1,popN);
            af::seq x;
            int modifier;
            for (int u = 0; u < netstructure.laynum; u++)
            {
                modifier = int(tensor->tensor[u].dims(1) - intraout.dims(0));
                if (modifier > 0)
                {
                    base = af::array(modifier,1,1,1);
                    base(modifier-1,0,0,span) = 1;
                    intraout = af::join(0,intraout,base);
                }
                //modifier = 1+(tensor(span, span, u, 0).dims(0) - intraout.dims(0));
                //if (modifier == 0) {modifier = 1;}
                //base = af::array(modifier,1,1,1);
                //base(modifier-1,0,0,0) = 1;
                //intraout = af::join(0,intraout,base);
                //cout << intraout.dims() << endl;
                //cout << "pop dims: " << population(span,span,u,span).dims() << endl;
                intraout = matmul(tensor->tensor[u](span, span, 0, 0), intraout);
                //cout << intraout.dims() << endl;
                intraout = act(intraout,netstructure.layers[u].func);
                /*
                if(netstructure.layers[u].Ncount < netstructure.max.first)
                {
                    intraout(seq(netstructure.layers[u].Ncount,netstructure.max.first-1),span,span,0) = 0;
                }*/

            }
            return intraout(seq(netstructure.outs),span,span,0);
        }
    }
    af::array run(int num)
    {
        af::array x = compute(testIn[num], false);
        af_print(x);
        return x;
    }
    void errorcalc(bool batched = false, int batchsize = 0,int batch = 0)
    {
        error = 0;
        af::array correct;
        af::array sum;
        af::array out;
        if (batched)
        {

            int shift =  batchsize * (batch);
            for (int i = 0; i < batchsize; i++)
            {
                out = compute(trainIn[i+shift]);
                correct = tile(trainOut[i+shift],1,1,1,popN);
                sum = (correct - out);
                sum = pow(sum,2);
                sum = af::sum(sum,0);
                error += sum;
            }
            error = error * 2;
            error = error / batchsize;
        }
        else
        {
            for (int i = 0; i < trainIn.size(); i++)
            {
                out = compute(trainIn[i]);
                correct = tile(trainOut[i],1,1,1,popN);
                sum = (correct - out);
                sum = pow(sum,2);
                sum = af::sum(sum,0);
                error += sum;
            }
            error = error * 2;
            error = error / trainIn.size();
        }
    }

    //genetic algo specific
    void evolve(float maxmute, float topFrac,int generations,bool debug = false, bool batched = false, int batchsize = 1)
    {
        if (batched)
        {
            for (int i = 0; i < int(trainIn.size()/batchsize); i++)
            {
                for (int o = 0; o < generations; o++)
                {
                    errorcalc(batched,batchsize,i);
                    selection(topFrac);
                    if(debug){cout << "gen: " << (o+1) << " batch: " << (i+1) << endl; optimalerror();}
                    //af_print(population(0,0,0,0));
                    reproduce(topFrac);
                    mutate(maxmute);
                    tensor->setnetequal(optimal);
                }
            }
        } else
        {
            for (int o = 0; o < generations; o++)
            {
                errorcalc();
                optimalerror();
                selection(topFrac);
                if(debug){cout << "gen: " << (1+o) << endl; optimalerror();}
                reproduce(topFrac);
                mutate(maxmute);
                tensor->setnetequal(optimal);
            }
        }

    }
    void mutate(float maxmute)
    {
        rengine.setSeed(randint());
        auto * mutagen = new NeuralTensor(netstructure,rengine,popN,maxmute);
        mutagen->setnetequal(0.0,0);
        tensor->add(mutagen);
        tensor->setnetequal(optimal);
    }
    void selection(float topFrac)
    {
        //indexing
        af::array index = af::array(1,1,1,popN);
        gfor(seq i, popN)
        {
            index(span,span,span,i) = i;
        }
        af::array strong;

        af::sort(strong,index, error,3, true);
        error = strong;
        //af_print(strong(0));
        //final selection
        int topnum = topFrac * popN;
        index = index(span,span,span,seq(topnum));
        //af_print(index(0));
        //af_print(error(0));
        auto * fittest = new NeuralTensor(netstructure,rengine,popN,0);
        index = moddims(index,topnum);
        fittest->tensor = tensor->extractNN(index);
        fittest->tensor = fittest->extractNN(seq(topnum));
        optimal = new NeuralTensor(fittest->extractNN());
        tensor = fittest;
        //af_print(population(0,0,0,0));
    }
    void reproduce(float topfrac)
    {
        for (auto & i : tensor->tensor)
        {
            i = tile(i, 1, 1, 1, int(1 / topfrac));
        }
    }

    //Gradient Decent Specific

};