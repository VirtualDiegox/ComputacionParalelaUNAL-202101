#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <string>

using namespace std;
using namespace cv;

__global__ void grisPromedioCUDA(uint8_t * Ptr_src,uint8_t * Ptr_dst, int *bloques, int *height, int *width,int step){
    int index = threadIdx.x + blockIdx.x * blockDim.x; 
    int hilos = blockDim.x * (*bloques); 
    uint8_t pixel[3];
    int pixelsPerThread = (*height)*(*width) / hilos;
    //segun la cantidad de hilos dividimos las iteraciones por block-wise y se las pasamos a una estructura para pasar la info al hilo
    int inicioy = (int)((pixelsPerThread * index)/(*width) ) ;
    int finy = (int)((pixelsPerThread * (index + 1))/(*width));

    int iniciox = (int)((pixelsPerThread * index)%(*width));
    int finx = (int)((pixelsPerThread * (index+1))%(*width));
    
    //if (index == (hilos - 1)){
       // fin = (*height) - 1;
    //}
    int ancho = *width;
    int paso = step;
    
    for (int i = inicioy; i <= finy; i++){
        for (int j = ( i == inicioy ) ? iniciox : 0 ; j < ((i == finy)? finx : ancho); j++){
            //obtenemos valores RGB de la imagen
            pixel[0] = (uint8_t)(Ptr_src[i * paso * 3  + j * 3 + 0]); // B
            pixel[1] = (uint8_t)(Ptr_src[i * paso * 3 + j * 3 + 1]); // G
            pixel[2] = (uint8_t)(Ptr_src[i * paso * 3 + j * 3 + 2]); // R
            

            //hacemos la logica del filtro con los valores RGB
            uint8_t Grey = (pixel[0] + pixel[1] + pixel[2]) / 3;
            //asignamos el valor calculado al unico canal de la imagen a crear
           
            Ptr_dst[i * paso + j] = Grey;
        };
     
    };
    
}
__global__ void grisLumaCUDA(uint8_t * Ptr_src,uint8_t * Ptr_dst, int *bloques, int *height, int *width,int step){
    int index = threadIdx.x + blockIdx.x * blockDim.x; 
    int hilos = blockDim.x * (*bloques); 
    uint8_t pixel[3];
    int pixelsPerThread = (*height)*(*width) / hilos;
    //segun la cantidad de hilos dividimos las iteraciones por block-wise y se las pasamos a una estructura para pasar la info al hilo
    int inicioy = (int)((pixelsPerThread * index)/(*width) ) ;
    int finy = (int)((pixelsPerThread * (index + 1))/(*width));

    int iniciox = (int)((pixelsPerThread * index)%(*width));
    int finx = (int)((pixelsPerThread * (index+1))%(*width));
    
    
    //if (index == (hilos - 1)){
        //fin = (*height) - 1;
    //}
    int ancho = *width;
    int paso = step;
    
    for (int i = inicioy; i <= finy; i++){
        for (int j = ( i == inicioy ) ? iniciox : 0 ; j < ((i == finy)? finx : ancho); j++){
            //obtenemos valores RGB de la imagen
            pixel[0] = (uint8_t)(Ptr_src[i * paso * 3  + j * 3 + 0]); // B
            pixel[1] = (uint8_t)(Ptr_src[i * paso * 3 + j * 3 + 1]); // G
            pixel[2] = (uint8_t)(Ptr_src[i * paso * 3 + j * 3 + 2]); // R
            

            //hacemos la logica del filtro con los valores RGB
            uint8_t Grey = (pixel[0]*0.0722+pixel[1]*0.7152+pixel[2]*0.2126);
            //asignamos el valor calculado al unico canal de la imagen a crear
           
            Ptr_dst[i * paso + j] = Grey;
        };
     
    };
    
}

__global__ void sombrasDeGrisCUDA(uint8_t * Ptr_src,uint8_t * Ptr_dst, int *bloques, int *height, int *width,int step,int *capas){
    int index = threadIdx.x + blockIdx.x * blockDim.x; 
    int hilos = blockDim.x * (*bloques); 
    uint8_t pixel[3];
    if(*capas<2) *capas = 2;
    if(*capas>255) *capas = 255;
    int ConversionFactor = 255 / (*capas - 1);
    //segun la cantidad de hilos dividimos las iteraciones por block-wise y se las pasamos a una estructura para pasar la info al hilo
    int pixelsPerThread = (*height)*(*width) / hilos;
    
    int inicioy = (int)((pixelsPerThread * index)/(*width) ) ;
    int finy = (int)((pixelsPerThread * (index + 1))/(*width));

    int iniciox = (int)((pixelsPerThread * index)%(*width));
    int finx = (int)((pixelsPerThread * (index+1))%(*width));

    //if (index == (hilos - 1)){
        //finy = (*height) - 1;
    //}
    
    int ancho = *width;
    int paso = step;
    
    for (int i = inicioy; i <= finy; i++){

        for (int j = ( i == inicioy ) ? iniciox : 0 ; j < ((i == finy)? finx : ancho); j++){


            //obtenemos valores RGB de la imagen
            pixel[0] = (uint8_t)(Ptr_src[i * paso * 3  + j * 3 + 0]); // B
            pixel[1] = (uint8_t)(Ptr_src[i * paso * 3 + j * 3 + 1]); // G
            pixel[2] = (uint8_t)(Ptr_src[i * paso * 3 + j * 3 + 2]); // R
            

            //hacemos la logica del filtro con los valores RGB
            uint8_t AverageValue = (pixel[0]+pixel[1]+pixel[2])/3;
            uint8_t Grey = (int)((AverageValue / ConversionFactor) + 0.5) * ConversionFactor;
            //asignamos el valor calculado al unico canal de la imagen a crear
           
            Ptr_dst[i * paso + j] = Grey;
        };
    };  
}




void grisPromedio(Mat Ptr_src, Mat Ptr_dst, int hilos, int bloques)
{

    struct timeval *tval_before, *tval_after, *tval_result;
    tval_before = (struct timeval *)malloc(sizeof(struct timeval));
    tval_after = (struct timeval *)malloc(sizeof(struct timeval));
    tval_result = (struct timeval *)malloc(sizeof(struct timeval));

    int cn = Ptr_src.channels();//Numero de canales    
   
    cv::cuda::GpuMat d_src{ Ptr_src.rows, Ptr_src.cols, 1000};
    cv::cuda::GpuMat d_dst{ Ptr_src.rows, Ptr_src.cols, 1000};
    cv::cuda::createContinuous(Ptr_src.rows, Ptr_src.cols,CV_8UC3,d_src);
    cv::cuda::createContinuous(Ptr_src.rows, Ptr_src.cols,CV_8UC1,d_dst);
    
    d_src.upload(Ptr_src);
    d_dst.upload(Ptr_dst);


    int height = Ptr_src.rows;
    int width = Ptr_dst.cols;
    int step = (int)Ptr_src.step/(int)sizeof(uint8_t);
    int size = Ptr_src.rows * Ptr_src.step;
    

    //CUDA
   
    int *d_height, *d_bloques, *d_width,*d_step;

    
    cudaMalloc((void **)&d_height, sizeof(int));
    cudaMalloc((void **)&d_width, sizeof(int));
    cudaMalloc((void **)&d_bloques, sizeof(int));
    cudaMalloc((void **)&d_step, sizeof(int));


    cudaMemcpy(d_height, &height, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_width, &width, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bloques, &bloques, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_step, &step, sizeof(int), cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    gettimeofday(tval_before, NULL);
    grisPromedioCUDA<<<bloques, hilos,0,stream>>>((uint8_t*)d_src.data, (uint8_t*)d_dst.data, d_bloques, d_height, d_width,d_dst.step);
    cudaStreamSynchronize(stream);
    gettimeofday(tval_after, NULL);
    timersub(tval_after, tval_before, tval_result);
    printf("%ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
    
    d_dst.download(Ptr_dst);
    cudaStreamDestroy(stream);
    cudaFree(d_bloques);
    cudaFree(d_height);
    cudaFree(d_width);
    cudaFree(d_step);
}

void grisLuma(Mat Ptr_src, Mat Ptr_dst, int hilos, int bloques)
{

    struct timeval *tval_before, *tval_after, *tval_result;
    tval_before = (struct timeval *)malloc(sizeof(struct timeval));
    tval_after = (struct timeval *)malloc(sizeof(struct timeval));
    tval_result = (struct timeval *)malloc(sizeof(struct timeval));
    int cn = Ptr_src.channels();//Numero de canales    
    
    cv::cuda::GpuMat d_src{ Ptr_src.rows, Ptr_src.cols, 1000};
    cv::cuda::GpuMat d_dst{ Ptr_src.rows, Ptr_src.cols, 1000};
    cv::cuda::createContinuous(Ptr_src.rows, Ptr_src.cols,CV_8UC3,d_src);
    cv::cuda::createContinuous(Ptr_src.rows, Ptr_src.cols,CV_8UC1,d_dst);
    
    d_src.upload(Ptr_src);
    d_dst.upload(Ptr_dst);

    int height = Ptr_src.rows;
    int width = Ptr_dst.cols;
    int step = (int)Ptr_src.step/(int)sizeof(uint8_t);
    int size = Ptr_src.rows * Ptr_src.step;
    

    //CUDA
   
    int *d_height, *d_bloques, *d_width,*d_step;

    
    cudaMalloc((void **)&d_height, sizeof(int));
    cudaMalloc((void **)&d_width, sizeof(int));
    cudaMalloc((void **)&d_bloques, sizeof(int));
    cudaMalloc((void **)&d_step, sizeof(int));


    cudaMemcpy(d_height, &height, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_width, &width, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bloques, &bloques, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_step, &step, sizeof(int), cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    gettimeofday(tval_before, NULL);
    grisLumaCUDA<<<bloques, hilos,0,stream>>>((uint8_t*)d_src.data, (uint8_t*)d_dst.data, d_bloques, d_height, d_width,d_dst.step);
    cudaStreamSynchronize(stream);
    gettimeofday(tval_after, NULL);
    timersub(tval_after, tval_before, tval_result);
    printf("%ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
    
    d_dst.download(Ptr_dst);
    cudaStreamDestroy(stream);
    cudaFree(d_bloques);
    cudaFree(d_height);
    cudaFree(d_width);
    cudaFree(d_step);
}

void sombrasDeGris(Mat Ptr_src, Mat Ptr_dst, int hilos, int bloques,int capas)
{

    struct timeval *tval_before, *tval_after, *tval_result;
    tval_before = (struct timeval *)malloc(sizeof(struct timeval));
    tval_after = (struct timeval *)malloc(sizeof(struct timeval));
    tval_result = (struct timeval *)malloc(sizeof(struct timeval));
    int cn = Ptr_src.channels();//Numero de canales    
   
    cv::cuda::GpuMat d_src{ Ptr_src.rows, Ptr_src.cols, 1000};
    cv::cuda::GpuMat d_dst{ Ptr_src.rows, Ptr_src.cols, 1000};
    cv::cuda::createContinuous(Ptr_src.rows, Ptr_src.cols,CV_8UC3,d_src);
    cv::cuda::createContinuous(Ptr_src.rows, Ptr_src.cols,CV_8UC1,d_dst);
    
    d_src.upload(Ptr_src);
    d_dst.upload(Ptr_dst);

    int height = Ptr_src.rows;
    int width = Ptr_dst.cols;
    int step = (int)Ptr_src.step/(int)sizeof(uint8_t);
    int size = Ptr_src.rows * Ptr_src.step;
    

    //CUDA
   
    int *d_height, *d_bloques, *d_width,*d_step,*d_capas;

    
    cudaMalloc((void **)&d_height, sizeof(int));
    cudaMalloc((void **)&d_width, sizeof(int));
    cudaMalloc((void **)&d_bloques, sizeof(int));
    cudaMalloc((void **)&d_step, sizeof(int));
    cudaMalloc((void **)&d_capas, sizeof(int));



    cudaMemcpy(d_height, &height, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_width, &width, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bloques, &bloques, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_step, &step, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_capas, &capas, sizeof(int), cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    gettimeofday(tval_before, NULL);
    sombrasDeGrisCUDA<<<bloques, hilos,0,stream>>>((uint8_t*)d_src.data, (uint8_t*)d_dst.data, d_bloques, d_height, d_width,d_dst.step,d_capas);
    cudaStreamSynchronize(stream);
    gettimeofday(tval_after, NULL);
    timersub(tval_after, tval_before, tval_result);
    printf("%ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
    
    d_dst.download(Ptr_dst);
    cudaStreamDestroy(stream);
    cudaFree(d_bloques);
    cudaFree(d_height);
    cudaFree(d_width);
    cudaFree(d_step);
    cudaFree(d_capas);
}


int main(int argc, char *argv[])
{
    
    //declaramos argumentos de entradas
    char *nombre_src;
    nombre_src = (char *)malloc(sizeof(char) * 20);
    char *nombre_dst;
    nombre_dst = (char *)malloc(sizeof(char) * 20);
    int parametro_filtro, capas, hilos, bloques;
    //tomando argumentos por consola
    strcpy(nombre_src, argv[1]);
    strcpy(nombre_dst, argv[2]);
    parametro_filtro = stoi(argv[3]);
    if (parametro_filtro == 3 || parametro_filtro == 4)
    {
        capas = stoi(argv[4]);
        hilos = stoi(argv[5]);
        bloques = stoi(argv[6]);
    }
    else
    {
        hilos = stoi(argv[4]);
        bloques = stoi(argv[5]);
    }

    //caso en el que se quiera correr el filtro secuencialmente

    //declaramos variable para tomar el tiempo
    //struct timeval *tval_before, *tval_after, *tval_result;
    //tval_before = (struct timeval *)malloc(sizeof(struct timeval));
    //tval_after = (struct timeval *)malloc(sizeof(struct timeval));
    //tval_result = (struct timeval *)malloc(sizeof(struct timeval));
    //Leemos la imagen
    Mat imagen_src = imread(nombre_src);
    free(nombre_src);
    //Advertimos si no se encuentra la imagen
    if (imagen_src.empty())
    {
        printf(" Error opening image\n");
        return -1;
    }

    //Declaramos objeto sobre el cual se trabajara la imagen a crear
    Mat image_dst(imagen_src.rows, imagen_src.cols, CV_8UC1, Scalar(0));

    int height = imagen_src.rows;
    int width = imagen_src.cols;

    switch (parametro_filtro)
    {
    //filtro grisPromedio
    case 1:
        grisPromedio(imagen_src, image_dst, hilos, bloques);
        break;
    //filtro Luma
    case 2:
        
        grisLuma(imagen_src,image_dst, hilos, bloques);
        
        break;
    //filtro sombrasDeGris
    case 3:
        
        sombrasDeGris(imagen_src,image_dst, hilos, bloques,capas);
        
        break;
    //filtro granular
    case 4:
        //gettimeofday(tval_before, NULL);
        //granular(imagen_src,image_dst,capas);
        //gettimeofday(tval_after, NULL);
        //timersub(tval_after, tval_before, tval_result);
        //printf("%ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
        break;
    default:
        break;
    }
    //Guardamos la imagen
    imwrite(nombre_dst, image_dst);
    free(nombre_dst);
    return 1;

    //40 Multiprocessors 128 hilos
}
