#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>



__global__ void add(int* a,int* b,int* c){
    *c = *a + *b;
    printf("sumando");
}


int main(){
    printf("%s",CV_VERSION);
    int a,b,c;
    int *d_a, *d_b, *d_c;
    int size = sizeof(int);
    cudaMalloc((void**)&d_a,size);
    cudaMalloc((void**)&d_b,size);
    cudaMalloc((void**)&d_c,size);
    a = 4;
    b = 5;
    c = 0;
    cudaMemcpy(d_a,&a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,&b,size,cudaMemcpyHostToDevice);

    add<<<1,1>>>(d_a,d_b,d_c);

    cudaMemcpy(&c,d_c,size,cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    printf("%d\n",c);

    return 0;




}