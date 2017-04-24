// Threaded two-dimensional Discrete FFT transform
// YOUR NAME HERE
// ECE8893 Project 2

#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
#include <pthread.h>
#include "Complex.h"
#include "InputImage.h"
#include <cstring>
// You will likely need global variables indicating how
// many threads there are, and a Complex* that points to the
// 2d image being transformed.

using namespace std;



Complex* ImageData;
Complex After1D[1024*1024];
Complex Final1D[1024*1024];
Complex* W;
Complex* Weights;
int height;
int width;
int imageSize;
unsigned int N;
unsigned int numThreads;
int size;
pthread_barrier_t barrier;
pthread_mutex_t startMutex;
pthread_mutex_t endMutex;
int count;
int count2 = 16;
pthread_cond_t condition;
// Function to reverse bits in an unsigned integer
// This assumes there is a global variable N that is the
// number of points in the 1D transform.
unsigned ReverseBits(unsigned v)
{ //  Provided to students
  unsigned n = N; // Size of array (which is even 2 power k value)
  unsigned r = 0; // Return value
   
  for (--n; n > 0; n >>= 1)
    {
      r <<= 1;        // Shift return value
      r |= (v & 0x1); // Merge in next bit
      v >>= 1;        // Shift reversal value
    }
  return r;
}


void Transpose(Complex* h, int w, int sr) {

/*for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
     newArray[j*w+i] = h[i*w+j];
*/
for (int i = sr; (i - sr) < w/(int) numThreads; i++) {
    for (int j = 0; j < w; j++) {
        if ( i < j) {
            Complex newArr = h[i*w+j];
            h[i * w + j] = h[j*w + i];
            h[j*w+i] = newArr; 
        }
    }
}
}


void EndTranspose(Complex* ogArray, Complex* newArray ,int w, int h) {
cout << "Final Transpose" << endl;
for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
        newArray[j*w+i] = ogArray[i*w+j];
    }
}
} 

void WeightValue(Complex* w, int length) {

for (int i = 0; i < length/2; i++) {

Complex weight(cos(2*M_PI*i/length), -1*sin(2*M_PI*i/length));
Complex WM = weight.Mag();
w[i] = weight;
w[i + (length/2)] = w[i + (length/2)] - weight;

}
}




// GRAD Students implement the following 2 functions.
// Undergrads can use the built-in barriers in pthreads.

// Call MyBarrier_Init once in main
void MyBarrier_Init(pthread_barrier_t* b, unsigned numThreads)// you will likely need some parameters)
{
pthread_barrier_init(b, NULL, numThreads); 
}

// Each thread calls MyBarrier after completing the row-wise DFT
void MyBarrier(pthread_barrier_t* b) // Again likely need parameters
{
pthread_barrier_wait(b);
}
                    
void Transform1D(Complex* h, int N)
{
  // Implement the efficient Danielson-Lanczos DFT here.
  // "h" is an input/output parameter
  // "N" is the size of the array (assume even power of 2)
  for (int i = 0; i < N; i++) {
      int ir = ReverseBits(i);
      if (ir < i) {
          Complex reverse = h[ir];
          h[ir] = h[i];
          h[i] = reverse;
      }
  }
  for (int j = 2; j <= N; j *= 2) {
      for (int k = 0; k < N; k += j) {
          for (int l = k; l < k + j/2; l++) {
             Complex even = h[l];
             Complex odd = h[l + j/2];
             h[l] = even + W[(l*N/j) % N] * odd;
             h[l + j/2] = even - W[(l*N/j) % N] * odd;       
          }
      }
  }
}

void* Transform2DTHread(void* v)
{ // This is the thread startign point.  "v" is the thread number
  // Calculate 1d DFT for assigned rows
  // wait for all to complete
  // Calculate 1d DFT for assigned columns
  // Decrement active count and signal main if all complete
  InputImage Image1D("Tower.txt");
  unsigned long rowStarter = (unsigned long) v;
  unsigned int rows = N/numThreads; 
  unsigned rowNum = rowStarter * rows;
  Complex* rowArr = ImageData + rowNum * N;
  for (unsigned int i = 0; i < rows; i++) {
      Transform1D(rowArr, N);
      rowArr += N;
  }
  MyBarrier(&barrier);
  Transpose(ImageData, N, rowNum);
   if (count2 == 0) {
   for (unsigned int i = 0; i < N * N; i++) {
      After1D[i] = ImageData[i];
  } }
 
  MyBarrier(&barrier);
  rowArr = ImageData + rowNum * N;
  for (unsigned int j = 0; j < rows; j++) {
      Transform1D(rowArr, N);
      rowArr += N;
  }
  MyBarrier(&barrier);
  Transpose(ImageData, N, rowNum);
  pthread_mutex_lock(&startMutex);
  count--;
  pthread_mutex_unlock(&startMutex);
  if (count == 0) {
     pthread_mutex_lock(&endMutex);
     pthread_cond_signal(&condition);
     pthread_mutex_unlock(&endMutex);
  }
  return 0;
}

void Transform2D(const char* inputFN) 
{ // Do the 2D transform here.
  InputImage image(inputFN);  // Create the helper object for reading the image
  // Create the global pointer to the image array data
  // Create 16 threads
  // Wait for all threads complete
  // Write the transformed data
  N = image.GetWidth();
  ImageData = image.GetImageData();
  Complex Wvalues[N];
  WeightValue(Wvalues, N);
  W = Wvalues;
  pthread_mutex_init(&endMutex, 0);
  pthread_mutex_init(&startMutex, 0);
  pthread_cond_init(&condition, 0);
  pthread_mutex_init(&endMutex, 0);
  MyBarrier_Init(&barrier, numThreads);
  count = numThreads; 
  for (unsigned int i = 0; i < numThreads; i++) {
      pthread_t thread1;
      pthread_create(&thread1, 0, Transform2DTHread, (void*) i);
      count2--;
  }
  cout << "Completed calculations" << endl;
  pthread_cond_wait(&condition, &endMutex);
  EndTranspose(After1D, Final1D, N, N);
  image.SaveImageData("MyAfter1D.txt", Final1D, N, N);
  image.SaveImageData("MyAfter2D.txt", ImageData, N, N);
}

int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  // MPI initialization here
  numThreads = 16;
  Transform2D(fn.c_str()); // Perform the transform.
}  
  

  
