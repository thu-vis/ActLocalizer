//example.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;
using std::cout;

struct buffer_info {
    void *ptr;
    ssize_t itemsize;
    std::string format;
    ssize_t ndim;
    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;
}; 

py::array_t<double> reorder_func(py::array_t<double> input1) {
    py::buffer_info buf1 = input1.request();
 
    /* No pointer is passed, so NumPy will allocate the buffer */
    int n = sqrt(buf1.size);
    auto result = py::array_t<double>(n);
 
    py::buffer_info buf2 = result.request();
 
    double *ptr1 = (double *) buf1.ptr,
        *ptr2 = (double *) buf2.ptr;
    // for(int i=0;i<n;i++) {
    //     for(int j=0;j<n;j++) printf("%lf ", ptr1[i*n+j]);
    //     printf("\n");
    // }
    
    // printf("1\n");
    double **ans = new double*[1<<n];
    int **trace = new int*[1<<n];
    for(int i=0;i<(1<<n);i++) {
        ans[i] = new double[n];
        trace[i] = new int[n];
        for(int j=0;j<n;j++) {
            ans[i][j] = 10000000000;
            trace[i][j] = 0;
        }
    }

    // printf("2\n");
    ans[0][0] = 0;
    for(int i=0;i<(1<<n);i++)
      for(int j=0;j<n;j++) 
        for(int k=0;k<n;k++) {
            if((i & (1<<k)) != 0) continue;
            double q=0;
            if(i>0)q=ptr1[j*n+k];
            if(ans[i][j]+q<ans[i+(1<<k)][k]) {
                ans[i+(1<<k)][k]=ans[i][j]+q;
                trace[i+(1<<k)][k]=j;
            }
        }
    
    // printf("3\n");
    int last_s = 0;
    for(int i=0;i<n;i++)
      if(ans[(1<<n)-1][i]<ans[(1<<n)-1][last_s])
        last_s=i;
    //printf("%lf\n",ans[(1<<n)-1][last_s]);
    
    // printf("4\n");
    int now = (1<<n)-1, now_s=last_s;
    for(int i=n;i>=1;i--) {
        // printf("%d %d\n",now, now_s);
        ptr2[i-1]=now_s;
        int new_s=trace[now][now_s];
        now=now-(1<<now_s);
        now_s=new_s;
    }

    //printf("5\n");
    for(int i=0;i<(1<<n);i++) {
        delete ans[i];
        delete trace[i];
    }
    delete ans;
    delete trace;

    return result;
 }

PYBIND11_MODULE(reorder, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
 
    m.def("reorder_func", &reorder_func, "A function which adds two numbers");
}