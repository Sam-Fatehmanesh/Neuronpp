#include <vector>
#include <string>
#include <arrayfire.h>
#include <af/util.h>

using namespace std;

class Layerspecs
{
public:
    int incount;
    int Ncount;
    string func;
    af::array neurons;

    Layerspecs(int ncount,  string Func, int IN = 0)
    {
        Ncount = ncount;
        func = Func;
        incount = IN;
    }
};