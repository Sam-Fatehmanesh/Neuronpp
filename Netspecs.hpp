#include "layerspecs.hpp"

class NetSpecs {
public:
    int ins;
    vector<Layerspecs> layers;
    int laynum = 0;
    int outs;
    pair<int,int> max;
    explicit NetSpecs(int I,int Outs = 0)
    {
        outs = Outs;
        ins = I;
    }
    void addlayer(int ncount, const string& Func = "")
    {
        laynum = laynum + 1;
        if (layers.empty())
        {
            layers.emplace_back(ncount,Func,ins);
        }
        else
        {
            int temp = layers.size()-1;
            int x = layers[temp].Ncount;
            layers.emplace_back(ncount,Func,x);
        }


    }
    bool issparse()
    {
        //need to replace maxes
        float nonzeros = 0;
        for(auto& i : layers)
        {
            nonzeros += float(i.incount * i.Ncount);
        }

        float total = float(max.first) * float(max.second)  * float(laynum);

        return nonzeros / total < 0.5;
    }
};