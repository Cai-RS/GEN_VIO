/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libviso2.
Authors: Andreas Geiger

libviso2 is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or any later version.

libviso2 is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libviso2; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#ifndef _MSC_VER
  #include <stdint.h>
#else
  typedef __int8            int8_t;
  typedef __int16           int16_t;
  typedef __int32           int32_t;
  typedef __int64           int64_t;
  typedef unsigned __int8   uint8_t;
  typedef unsigned __int16  uint16_t;
  typedef unsigned __int32  uint32_t;
  typedef unsigned __int64  uint64_t;
#endif

#define endll endl << endl // double end line definition

typedef double FLOAT;      // double precision
//typedef float  FLOAT;    // single precision

class MatriX {

public:

  // constructor / deconstructor
  MatriX ();                                                  // init empty 0x0 matrix
  MatriX (const int32_t m,const int32_t n);                   // init empty mxn matrix
  MatriX (const int32_t m,const int32_t n,const FLOAT* val_); // init mxn matrix with values from array 'val'
  MatriX (const MatriX &M);                                   // creates deepcopy of M
  ~MatriX ();

  // assignment operator, copies contents of M
  MatriX& operator= (const MatriX &M);

  // copies submatrix of M into array 'val', default values copy whole row/column/matrix
  void getData(FLOAT* val_,int32_t i1=0,int32_t j1=0,int32_t i2=-1,int32_t j2=-1);

  // set or get submatrices of current matrix
  MatriX getMat(int32_t i1,int32_t j1,int32_t i2=-1,int32_t j2=-1);
  void   setMat(const MatriX &M,const int32_t i,const int32_t j);

  // set sub-matrix to scalar (default 0), -1 as end replaces whole row/column/matrix
  void setVal(FLOAT s,int32_t i1=0,int32_t j1=0,int32_t i2=-1,int32_t j2=-1);

  // set (part of) diagonal to scalar, -1 as end replaces whole diagonal
  void setDiag(FLOAT s,int32_t i1=0,int32_t i2=-1);

  // clear MatriX
  void zero();
  
  // extract columns with given index
  MatriX extractCols (std::vector<int> idx);

  // create identity MatriX
  static MatriX eye (const int32_t m);
  void          eye ();

  // create diagonal matrix with nx1 or 1xn MatriX M as elements
  static MatriX diag(const MatriX &M);
  
  // returns the m-by-n matrix whose elements are taken column-wise from M
  static MatriX reshape(const MatriX &M,int32_t m,int32_t n);

  // create 3x3 rotation matrices (convention: http://en.wikipedia.org/wiki/Rotation_matrix)
  static MatriX rotMatX(const FLOAT &angle);
  static MatriX rotMatY(const FLOAT &angle);
  static MatriX rotMatZ(const FLOAT &angle);

  // simple arithmetic operations
  MatriX  operator+ (const MatriX &M); // add matrix
  MatriX  operator- (const MatriX &M); // subtract matrix
  MatriX  operator* (const MatriX &M); // multiply with matrix
  MatriX  operator* (const FLOAT &s);  // multiply with scalar
  MatriX  operator/ (const MatriX &M); // divide elementwise by matrix (or vector)
  MatriX  operator/ (const FLOAT &s);  // divide by scalar
  MatriX  operator- ();                // negative matrix
  MatriX  operator~ ();                // transpose
  FLOAT   l2norm ();                   // euclidean norm (vectors) / frobenius norm (matrices)
  FLOAT   mean ();                     // mean of all elements in matrix

  // complex arithmetic operations
  static MatriX cross (const MatriX &a, const MatriX &b);    // cross product of two vectors
  static MatriX inv (const MatriX &M);                       // invert matrix M
  bool   inv ();                                             // invert this matrix
  FLOAT  det ();                                             // returns determinant of matrix
  bool   solve (const MatriX &M,FLOAT eps=1e-20);            // solve linear system M*x=B, replaces *this and M
  bool   lu(int32_t *idx, FLOAT &d, FLOAT eps=1e-20);        // replace *this by lower upper decomposition
  void   svd(MatriX &U,MatriX &W,MatriX &V);                 // singular value decomposition *this = U*diag(W)*V^T

  // print matrix to stream
  friend std::ostream& operator<< (std::ostream& out,const MatriX& M);

  // direct data access
  FLOAT   **val;
  int32_t   m,n;

private:

  void allocateMemory (const int32_t m_,const int32_t n_);
  void releaseMemory ();
  inline FLOAT pythag(FLOAT a,FLOAT b);

};

#endif // MATRIX_H
