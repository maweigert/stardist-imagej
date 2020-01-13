#include <stdarg.h>
#include <Python.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include <stdio.h>
#include <limits>
#include <vector>
#include <array>
#include <string>
#include <chrono>

#include "libqhullcpp/QhullFacet.h"
#include "libqhullcpp/QhullError.h"
#include "libqhullcpp/Coordinates.h"
#include "libqhullcpp/RboxPoints.h"
#include "libqhullcpp/QhullFacetList.h"
#include "libqhullcpp/QhullFacetSet.h"
#include "libqhullcpp/QhullPointSet.h"
#include "libqhullcpp/QhullRidge.h"
#include "libqhullcpp/Qhull.h"

#include "utils.h"

using namespace orgQhull;

#define DIM 3

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

#ifdef _OPENMP
#include <omp.h>
#endif


inline int round_to_int(float r) {
  return (int)lrint(r);
}

int _sum_buffer(const bool * const buffer, const int N){
  int res = 0;
  for (int i=0; i <N; ++i)
	res += buffer[i];
  return res;
}



inline unsigned char inside_halfspace(float z, float y, float x,
									  float Az, float Ay, float Ax,
									  float Bz, float By, float Bx,
									  float Cz, float Cy, float Cx){
  float M00, M01, M02;
  float M10, M11, M12;
  float M20, M21, M22;
  float det;

  M00 = Bz-Az; M01 = By-Ay; M02 = Bx-Ax;
  M10 = Cz-Az; M11 = Cy-Ay; M12 = Cx-Ax;
  M20 = z-Az;  M21 = y-Ay;  M22 = x-Ax;


  det =  M00 * (M11 * M22 - M21 * M12) - M01 * (M10 * M22 - M12 * M20) + M02 * (M10 * M21 - M11 * M20);

  return (det>=0);
}


inline unsigned char inside_tetrahedron(float z, float y, float x,
										float Rz, float Ry, float Rx,
										float Az, float Ay, float Ax,
										float Bz, float By, float Bx,
										float Cz, float Cy, float Cx){

  // // first do a bounding box check
  // float z1 = fmin(fmin(Rz,Az),fmin(Bz,Cz));
  // float z2 = fmax(fmax(Rz,Az),fmax(Bz,Cz));
  // float y1 = fmin(fmin(Ry,Ay),fmin(By,Cy));
  // float y2 = fmax(fmax(Ry,Ay),fmax(By,Cy));
  // float x1 = fmin(fmin(Rx,Ax),fmin(Bx,Cx));
  // float x2 = fmax(fmax(Rx,Ax),fmax(Bx,Cx));
  // if ((z<z1) || (z>z2) || (y<y1) || (y>y2) || (x<x1) || (x>x2))
  // 	return 0;
  // else
  // 	return 1;

  // if inside the bounding box, do a proper check

  return (inside_halfspace(z,y,x,
                           Az,Ay,Ax,
                           Bz,By,Bx,
                           Cz,Cy,Cx)
          &&
          inside_halfspace(z,y,x,
                           Rz,Ry,Rx,
                           Bz,By,Bx,
                           Az,Ay,Ax)
          &&
          inside_halfspace(z,y,x,
                           Rz,Ry,Rx,
                           Cz,Cy,Cx,
                           Bz,By,Bx)
          &&
          inside_halfspace(z,y,x,
                           Rz,Ry,Rx,
                           Az,Ay,Ax,
                           Cz,Cy,Cx));
}




inline unsigned char inside_polyhedron(const float z,const float y,const float x,
									   const float * center,
									   const float * polyverts,
									   const int * const faces,
									   const int n_rays, const int n_faces){


  for (int i  = 0; i < n_faces; ++i) {
	int iA = faces[3*i];
	int iB = faces[3*i+1];
	int iC = faces[3*i+2];

	float Rz = center[0];
	float Ry = center[1];
	float Rx = center[2];

	float Az = polyverts[3*iA];
	float Ay = polyverts[3*iA+1];
	float Ax = polyverts[3*iA+2];

	float Bz = polyverts[3*iB];
	float By = polyverts[3*iB+1];
	float Bx = polyverts[3*iB+2];

	float Cz = polyverts[3*iC];
	float Cy = polyverts[3*iC+1];
	float Cx = polyverts[3*iC+2];

	if (inside_tetrahedron(z, y, x,
						   Rz, Ry, Rx,
						   Az,  Ay,  Ax,
						   Bz,  By,  Bx,
						   Cz,  Cy,  Cx))
	  return 1;

  }
  return 0;

}



inline unsigned char inside_polyhedron_kernel(const float z,
											  const float y,
											  const float x,
											  const float * center,
											  const float * polyverts,
											  const int * const faces,
											  const int n_rays,
											  const int n_faces){


  for (int i  = 0; i < n_faces; ++i) {
	int iA = faces[3*i];
	int iB = faces[3*i+1];
	int iC = faces[3*i+2];

	float Az = polyverts[3*iA];
	float Ay = polyverts[3*iA+1];
	float Ax = polyverts[3*iA+2];

	float Bz = polyverts[3*iB];
	float By = polyverts[3*iB+1];
	float Bx = polyverts[3*iB+2];

	float Cz = polyverts[3*iC];
	float Cy = polyverts[3*iC+1];
	float Cx = polyverts[3*iC+2];

	if (!inside_halfspace(z, y, x,
                          Az,  Ay,  Ax,
                          Bz,  By,  Bx,
                          Cz,  Cy,  Cx))
	  return 0;

  }
  return 1;

}


inline float tetrahedron_volume(float Rz, float Ry, float Rx,
								float Az, float Ay, float Ax,
								float Bz, float By, float Bx,
								float Cz, float Cy, float Cx){

  float M00, M01, M02;
  float M10, M11, M12;
  float M20, M21, M22;
  float det;

  M00 = Bz-Az; M01 = By-Ay; M02 = Bx-Ax;
  M10 = Cz-Az; M11 = Cy-Ay; M12 = Cx-Ax;
  M20 = Rz-Az;  M21 = Ry-Ay;  M22 = Rx-Ax;


  det =  M00 * (M11 * M22 - M21 * M12) - M01 * (M10 * M22 - M12 * M20) + M02 * (M10 * M21 - M11 * M20);


  return det/6.f;
}



inline float polyhedron_volume(const float * const dist,
                               const float * const verts,
                               const int * const faces,
                               const int n_rays, const int n_faces){


  float vol = 0.f;

  for (int i  = 0; i < n_faces; ++i) {
	int iA = faces[3*i];
	int iB = faces[3*i+1];
	int iC = faces[3*i+2];


	float Az = dist[iA]*verts[3*iA];
	float Ay = dist[iA]*verts[3*iA+1];
	float Ax = dist[iA]*verts[3*iA+2];

	float Bz = dist[iB]*verts[3*iB];
	float By = dist[iB]*verts[3*iB+1];
	float Bx = dist[iB]*verts[3*iB+2];

	float Cz = dist[iC]*verts[3*iC];
	float Cy = dist[iC]*verts[3*iC+1];
	float Cx = dist[iC]*verts[3*iC+2];

	vol += tetrahedron_volume(0, 0, 0,
							  Az, Ay, Ax,
							  Bz, By, Bx,
							  Cz, Cy, Cx);

  }

  return vol;
}

inline void polyhedron_centroid(const float * const dist,
								const float * const verts,
								const int * const faces,
								const int n_rays, const int n_faces,
								float * centroid){


  float vol = 0.f;
  float Rz = 0.f;
  float Ry = 0.f;
  float Rx = 0.f;

  for (int i  = 0; i < n_faces; ++i) {
	int iA = faces[3*i];
	int iB = faces[3*i+1];
	int iC = faces[3*i+2];


	float Az = dist[iA]*verts[3*iA];
	float Ay = dist[iA]*verts[3*iA+1];
	float Ax = dist[iA]*verts[3*iA+2];

	float Bz = dist[iB]*verts[3*iB];
	float By = dist[iB]*verts[3*iB+1];
	float Bx = dist[iB]*verts[3*iB+2];

	float Cz = dist[iC]*verts[3*iC];
	float Cy = dist[iC]*verts[3*iC+1];
	float Cx = dist[iC]*verts[3*iC+2];

	float curr_vol = tetrahedron_volume(0, 0, 0,
                                        Az, Ay, Ax,
                                        Bz, By, Bx,
                                        Cz, Cy, Cx);
    Rz += .25 * (Az+Bz+Cz)*curr_vol;
    Ry += .25 * (Ay+By+Cy)*curr_vol;
    Rx += .25 * (Ax+Bx+Cx)*curr_vol;

	vol += curr_vol;

  }

  centroid[0] = vol>1.e-10?Rz/vol:0.f;
  centroid[1] = vol>1.e-10?Ry/vol:0.f;
  centroid[2] = vol>1.e-10?Rx/vol:0.f;


}


inline float bounding_radius_outer(const float * const dist, const int n_rays){

  float r = 0;
  for (int i=0; i<n_rays; i++)
	r = fmax(r,dist[i]);

  return r;
}

inline float bounding_radius_inner(const float * const dist,
                                   const float * const verts,
                                   const int * const faces,
                                   const int n_rays, const int n_faces){

  float r = INFINITY;

  for (int i  = 0; i < n_faces; ++i) {
	int iA = faces[3*i];
	int iB = faces[3*i+1];
	int iC = faces[3*i+2];


	float Az = dist[iA]*verts[3*iA];
	float Ay = dist[iA]*verts[3*iA+1];
	float Ax = dist[iA]*verts[3*iA+2];

	float Bz = dist[iB]*verts[3*iB];
	float By = dist[iB]*verts[3*iB+1];
	float Bx = dist[iB]*verts[3*iB+2];

	float Cz = dist[iC]*verts[3*iC];
	float Cy = dist[iC]*verts[3*iC+1];
	float Cx = dist[iC]*verts[3*iC+2];

	// difference vector
	float pz = Bz-Az, py = By-Ay, px = Bx-Ax;
	float qz = Cz-Az, qy = Cy-Ay, qx = Cx-Ax;

	// normal vector
	// FIME check the definition of cross product!
	float Nz =  (px*qy-py*qx);
	float Ny = -(px*qz-pz*qx);
	float Nx =  (py*qz-pz*qy);
	float normz = 1.f/(sqrt(Nz*Nz+Ny*Ny+Nx*Nx)+1.e-10);
	Nz *= normz;
	Ny *= normz;
	Nx *= normz;

	float distance  = Az*Nz+Ay*Ny+Ax*Nx;

	r = fmin(r,distance);

  }
  return r;

}

// computes the outer radius in "isotropic" coordinates
inline float bounding_radius_outer_isotropic(const float * const dist,
											 const float * const verts,
											 const int n_rays,
											 const float * aniso){

  float r_squared_max = 0;
  for (int i=0; i<n_rays; i++){
	float z = aniso[0]*dist[i]*verts[3*i];
	float y = aniso[1]*dist[i]*verts[3*i+1];
	float x = aniso[2]*dist[i]*verts[3*i+2];

	float r_squared = z*z+y*y+x*x;

	r_squared_max = fmax(r_squared, r_squared_max);
  }

  return sqrt(r_squared_max);
}

inline float bounding_radius_inner_isotropic(const float * const dist,
											 const float * const verts,
											 const int * const faces,
											 const int n_rays,
											 const int n_faces,
											 const float * aniso){

  float r_min = INFINITY;

  for (int i  = 0; i < n_faces; ++i) {
	int iA = faces[3*i];
	int iB = faces[3*i+1];
	int iC = faces[3*i+2];


	float Az = aniso[0]*dist[iA]*verts[3*iA];
	float Ay = aniso[1]*dist[iA]*verts[3*iA+1];
	float Ax = aniso[2]*dist[iA]*verts[3*iA+2];

	float Bz = aniso[0]*dist[iB]*verts[3*iB];
	float By = aniso[1]*dist[iB]*verts[3*iB+1];
	float Bx = aniso[2]*dist[iB]*verts[3*iB+2];

	float Cz = aniso[0]*dist[iC]*verts[3*iC];
	float Cy = aniso[1]*dist[iC]*verts[3*iC+1];
	float Cx = aniso[2]*dist[iC]*verts[3*iC+2];

	// difference vector
	float pz = Bz-Az, py = By-Ay, px = Bx-Ax;
	float qz = Cz-Az, qy = Cy-Ay, qx = Cx-Ax;

	// normal vector
	float Nz =  (px*qy-py*qx);
	float Ny =  (pz*qx-px*qz);
	float Nx =  (py*qz-pz*qy);
	float normz = 1.f/(sqrt(Nz*Nz+Ny*Ny+Nx*Nx)+1.e-10);
	Nz *= normz;
	Ny *= normz;
	Nx *= normz;

	float r  = Az*Nz+Ay*Ny+Ax*Nx;

	r_min = fmin(r_min,r);

  }
  return r_min;

}

inline float intersect_sphere(const float r1, const float * const p1,
                              const float r2, const float * const p2){

  float dz = (p1[0]-p2[0]);
  float dy = (p1[1]-p2[1]);
  float dx = (p1[2]-p2[2]);
  float d = sqrt(dz*dz+dy*dy+dx*dx);

  if (d>(r1+r2))
	return 0;

  float rmin = fmin(r1,r2), rmax = fmax(r1,r2);
  if (rmax>d+rmin)
	return M_PI*4.f/3*rmin*rmin*rmin;


  float t = (r1+r2-d)/2/d;
  float h1 = (r2-r1+d)*t;
  float h2 = (r1-r2+d)*t;
  float v1 = M_PI/3*h1*h1*(3*r1-h1);
  float v2 = M_PI/3*h2*h2*(3*r2-h2);
  return v1+v2;
}


inline float intersect_sphere_isotropic(const float r1, const float * const p1,
                                        const float r2, const float * const p2, const float * const anisotropy){

  float dz = anisotropy[0]*(p1[0]-p2[0]);
  float dy = anisotropy[1]*(p1[1]-p2[1]);
  float dx = anisotropy[2]*(p1[2]-p2[2]);
  float d = sqrt(dz*dz+dy*dy+dx*dx);
  float rmin = fmin(r1,r2), rmax = fmax(r1,r2);

  if (d>(r1+r2))
	return 0;

  if (rmax>=d+rmin-1.e-10)
	return M_PI*4.f/3*rmin*rmin*rmin;


  float t = (r1+r2-d)/2/d;
  float h1 = (r2-r1+d)*t;
  float h2 = (r1-r2+d)*t;
  float v1 = M_PI/3*h1*h1*(3*r1-h1);
  float v2 = M_PI/3*h2*h2*(3*r2-h2);

  // printf("r1: %.2f \t r2 = %.2f \n", r1, r2);
  // printf("dz: %.2f \t dy = %.2f \t dx = %.2f \n", dz,dy,dx);

  return (v1+v2)/(anisotropy[0]*anisotropy[1]*anisotropy[2]);
}


inline float intersect_bbox(const int * box1, const int* box2){


  float wz = fmax(0, fmin(box1[1], box2[1]) - fmax(box1[0], box2[0]));
  float wy = fmax(0, fmin(box1[3], box2[3]) - fmax(box1[2], box2[2]));
  float wx = fmax(0, fmin(box1[5], box2[5]) - fmax(box1[4], box2[4]));

  return wx*wy*wz;

}

// calculates the bounding box (and optionally the vertices) of a star-polyhedron
// both left and right boundaries are inclusive, i.e. part of the polyhedron
inline void polyhedron_bbox(const float * const dist,
							const float * const center,
							const float * const verts,
							const int n_rays, int * bbox){

  int z1 = std::numeric_limits<int>::max(),z2 = -1;
  int y1 = std::numeric_limits<int>::max(),y2 = -1;
  int x1 = std::numeric_limits<int>::max(),x2 = -1;


  for (int j = 0; j < n_rays; ++j) {
    float z = center[0] + dist[j]*verts[3*j];
    float y = center[1] + dist[j]*verts[3*j+1];
    float x = center[2] + dist[j]*verts[3*j+2];

    z1 = std::min(z1,round_to_int(z));
    z2 = std::max(z2,round_to_int(z));

    y1 = std::min(y1,round_to_int(y));
    y2 = std::max(y2,round_to_int(y));

    x1 = std::min(x1,round_to_int(x));
    x2 = std::max(x2,round_to_int(x));

  }
  bbox[0] = z1;
  bbox[1] = z2;
  bbox[2] = y1;
  bbox[3] = y2;
  bbox[4] = x1;
  bbox[5] = x2;
}

// computes the polyhedron vertices
inline void polyhedron_polyverts(const float * const dist,
                                 const float * const center,
                                 const float * const verts,
                                 const int n_rays, float * polyverts){


  for (int j = 0; j < n_rays; ++j) {
	float z = center[0] + dist[j]*verts[3*j];
	float y = center[1] + dist[j]*verts[3*j+1];
	float x = center[2] + dist[j]*verts[3*j+2];

	polyverts[3*j]   = z;
	polyverts[3*j+1] = y;
	polyverts[3*j+2] = x;
  }
}

void render_polyhedron(const float * const dist, const float * const center,
					   const int * const bbox, const float * const polyverts,
					   const int * const faces, const int n_rays, const int n_faces,
					   bool * rendered, const int Nz, const int Ny, const int Nx){

  // printf("rendering polygon of bbox size %d %d %d \n",Nz,Ny,Nx);

  // #pragma omp parallel for
  for (int z=0; z <Nz; ++z) {
	for (int y=0; y <Ny; ++y) {
	  for (int x=0; x <Nx; ++x) {

		rendered[x+y*Nx+z*Nx*Ny] = inside_polyhedron(z+bbox[0], y+bbox[2],x+bbox[4],
													 center, polyverts,
													 faces, n_rays, n_faces);
	  }
	}
  }

}

int overlap_render_polyhedron(const float * const dist, const float * const center,
                              const int * const bbox, const float * const polyverts,
                              const int * const faces, const int n_rays,
                              const int n_faces, const bool * const rendered,
                              const int Nz, const int Ny, const int Nx,
                              const float overlap_maximal){

  // printf("rendering polygon of bbox size %d %d %d \n",Nz,Ny,Nx);

  int res = 0;

  // #pragma omp parallel for reduction(+:res)
  for (int z=0; z <Nz; ++z) {
	for (int y=0; y <Ny; ++y) {
	  for (int x=0; x <Nx; ++x) {
		const int pz = z+bbox[0];
		const int py = y+bbox[2];
		const int px = x+bbox[4];

		res += ((rendered[x+y*Nx+z*Nx*Ny]) && (inside_polyhedron(pz, py, px ,center, polyverts, faces, n_rays, n_faces)));

        if (res>overlap_maximal){
          return res;
        }
	  }
	}
  }
  return res;
}

int overlap_render_polyhedron_kernel(const float * const dist,
                                     const float * const center,
                                     const int * const bbox,
                                     const float * const polyverts,
                                     const int * const faces, const int n_rays,
                                     const int n_faces,
                                     const bool * const rendered,
                                     const int Nz, const int Ny, const int Nx){

  // printf("bbox %d %d   %d %d  %d %d \n",bbox[0],bbox[1],bbox[2],bbox[3],bbox[4],bbox[5]);
  // printf("dist %.2f %.2f %.2f \n",dist[0],dist[1],dist[2]);
  // printf("center %.2f %.2f %.2f \n",center[0],center[1],center[2]);
  // printf("polyverts %.2f %.2f %.2f \n",polyverts[0],polyverts[1],polyverts[2]);
  // printf("SUM %d\n",_sum_buffer(rendered, Nx*Ny*Nz));


  int res = 0;

  // #pragma omp parallel for reduction(+:res)
  for (int z=0; z <Nz; ++z) {
	for (int y=0; y <Ny; ++y) {
	  for (int x=0; x <Nx; ++x) {

		const int pz = z+bbox[0];
		const int py = y+bbox[2];
		const int px = x+bbox[4];

		// if ((inside_polyhedron_kernel(pz, py, px, center, polyverts, faces, n_rays, n_faces)) && !(inside_polyhedron(pz, py, px, center, polyverts, faces, n_rays, n_faces)))
		//   printf("YIKKKKKKKES   %.1f ,%.1f ,%.1f, %d ,%d ,%d\n", center[0], center[1], center[2], pz, py, px);

		res += ((rendered[x+y*Nx+z*Nx*Ny]) && (inside_polyhedron_kernel(pz, py, px, center, polyverts, faces, n_rays, n_faces)));


	  }
	}
  }

  return res;
}

// volume of intersection of halfspaces via Qhull
inline float qhull_volume_halfspace_intersection(const double * halfspaces,
									const double * interior_point,
                                                 const int nhalfspaces,
                                                 const float err_value){

  // convert to std::vector which is what qhull expects
  std::vector<double> int_point(interior_point, interior_point+DIM);

  // intersect halfspaces
  Qhull q;
  q.setFeasiblePoint(Coordinates(int_point));
    
  try{
  	q.runQhull("halfspaces", DIM+1, nhalfspaces, halfspaces, "H");
  }
  catch(QhullError &e){
  	// e.errorCode==6023 for not valid feasible point
	// std::cout << "halfspace intersection"<<std::endl;
	// std::cout <<e.what()<< std::endl;
    // for (int i = 0; i < nhalfspaces; ++i) 
    //   printf("%.2f %.2f %.2f %.2f\n",halfspaces[4*i+0],halfspaces[4*i+1],halfspaces[4*i+2],halfspaces[4*i+3]);
    // printf("nHalfsspaces %d\n", nhalfspaces);
    
    
	return err_value;
  }
  // construct intersection points
  // see https://github.com/scipy/scipy/blob/master/scipy/spatial/qhull.pyx#L2724
  // intersections = dual_equations[:, :-1]/-dual_equations[:, -1:] + interior_points

  std::vector<std::array<double,DIM>> intersections;


  auto facetlist = q.facetList();
  for (auto itr = facetlist.begin(); itr != facetlist.end(); ++itr){

  	std::array<coordT,DIM> inter;
  	QhullHyperplane plane = (*itr).hyperplane();

  	for (int i = 0; i < DIM; ++i)
  	  inter[i] = -plane[i]/plane.offset() + interior_point[i];

  	intersections.push_back(inter);
  }


  // get pointer
  double * pinter = (coordT *)(&intersections[0]);
  int npoints = intersections.size();

  // now the convex hull of the intersections
  try{
  	Qhull qvert("convex hull", DIM, npoints, pinter, "");
  	return qvert.volume();
  }
  catch(QhullError &e){
	// std::cout << "convex hull of kernel intersection"<<std::endl;
	// std::cout << e.what()<<std::endl;
  	return err_value;
  }

  return err_value;
}

// return halfspace of a single triangle
inline std::array<double,DIM+1> build_halfspace(const float * const A,
												const float * const B,
												const float * const C){
  std::array<double,DIM+1> halfspace;

  float Az = A[0], Ay = A[1], Ax = A[2];
  float Bz = B[0], By = B[1], Bx = B[2];
  float Cz = C[0], Cy = C[1], Cx = C[2];
  // the vectors P = B-A  and  Q = C-A
  float Pz = Bz-Az, Py = By-Ay, Px = Bx-Ax;
  float Qz = Cz-Az, Qy = Cy-Ay, Qx = Cx-Ax;
  // normal vector
  float Nz =   -(Py*Qx-Px*Qy);
  float Ny =   -(Px*Qz-Pz*Qx);
  float Nx =   -(Pz*Qy-Py*Qz);
  halfspace[0] = Nz;
  halfspace[1] = Ny;
  halfspace[2] = Nx;
  halfspace[3] = -(Az*Nz+Ay*Ny+Ax*Nx);
  return halfspace;
}

// create the halfspaces of the convex hull of a polyhedron
inline std::vector<std::array<double,DIM+1>>
halfspaces_convex(const float * const polyverts,
				  const int n_rays){

  // copy vertices to double type;
  std::vector<double> points(DIM*n_rays);
  for (int i = 0; i < DIM*n_rays; ++i)
	points[i] = polyverts[i];

  // get convex hull
  Qhull qvert("convex hull", DIM, n_rays, points.data(), "");

  //get halfspaces
  std::vector<std::array<double,DIM+1>> halfspaces;
  std::array<double,DIM+1> hs;

  auto facetlist = qvert.facetList();
  for (auto itr = facetlist.begin(); itr != facetlist.end(); ++itr){
	QhullHyperplane plane = (*itr).hyperplane();
	for (int i = 0; i < DIM; ++i)
	  hs[i] = plane[i];

	hs[DIM] = plane.offset();
	halfspaces.push_back(hs);
  }

  return halfspaces;

}

// create the halfspaces of the kernel of a polyhedron
inline std::vector<std::array<double,DIM+1>>
halfspaces_kernel(const float * const polyverts,
				  const int * const faces, const int n_faces){

  std::vector<std::array<double,DIM+1>> halfspaces;

  for (int i = 0; i < n_faces; ++i) {
	int iA = faces[3*i];
	int iB = faces[3*i+1];
	int iC = faces[3*i+2];

	halfspaces.push_back(build_halfspace(&polyverts[3*iA],
										 &polyverts[3*iB],
										 &polyverts[3*iC]));
  }

  return halfspaces;

}

inline int point_in_halfspaces(const float z, const float y, const float x,
							   std::vector<std::array<double,DIM+1>> halfspaces){

  for (auto hs = halfspaces.begin(); hs != halfspaces.end(); hs++) {
	if ((*hs)[0]*z + (*hs)[1]*y +  (*hs)[2]*x + (*hs)[3] >0)
	  return 0;
  }

  return 1;
}


inline float qhull_overlap_kernel(
						 const float * const polyverts1, const float * const center1,
						 const float * const polyverts2, const float * const center2,
                         const int * const faces,
                         const int n_rays,
                         const int n_faces, const int n_step=1){

  // build halfspaces
  std::vector<std::array<double,DIM+1>> halfspaces;

  for (int i = 0; i < n_faces; i+=n_step) {
	int iA = faces[3*i];
	int iB = faces[3*i+1];
	int iC = faces[3*i+2];

	halfspaces.push_back(build_halfspace(&polyverts1[3*iA],
                                         &polyverts1[3*iB],
										 &polyverts1[3*iC]));

	halfspaces.push_back(build_halfspace(&polyverts2[3*iA],
										 &polyverts2[3*iB],
										 &polyverts2[3*iC]));

  }

  double interior_point[DIM];

  interior_point[0] = .5*(center1[0]+center2[0]);
  interior_point[1] = .5*(center1[1]+center2[1]);
  interior_point[2] = .5*(center1[2]+center2[2]);


  return qhull_volume_halfspace_intersection(
  											 (double *)&halfspaces[0],
  											 interior_point,
  											 halfspaces.size(),
                                             0.f // err_value
                                             );

}


inline float qhull_overlap_convex_hulls(
                                        const float * const polyverts1, const float * const center1,
                                        const float * const polyverts2, const float * const center2,
                                        const int * const faces, const int n_rays, const int n_faces){


  try{
	// copy polyverts to double array
	std::vector<std::array<double,DIM>> verts1(n_rays), verts2(n_rays);

	for (int i = 0; i < n_rays; ++i) {
	  for (int j = 0; j < DIM; ++j) {
		verts1[i][j] = polyverts1[DIM*i+j];
		verts2[i][j] = polyverts2[DIM*i+j];
	  }
	}

	// get pointer
	double pcenter1[] = {center1[0], center1[1], center1[2]};
	double pcenter2[] = {center2[0], center2[1], center2[2]};
    
	// build convex hulls of the polygon
	Qhull qvert1("convex hull", DIM, n_rays, (double *)&verts1[0], "");
	Qhull qvert2("convex hull", DIM, n_rays, (double *)&verts2[0], "");

	// build halfspaces from them

	std::vector<double> halfspaces;

	auto facetlist1 = qvert1.facetList();
	for (auto itr = facetlist1.begin(); itr != facetlist1.end(); ++itr){
	  QhullHyperplane plane = (*itr).hyperplane();
	  for (int i = 0; i < DIM; ++i)
		halfspaces.push_back(plane[i]);
	  halfspaces.push_back(plane.offset());

	}

	auto facetlist2 = qvert2.facetList();
	for (auto itr = facetlist2.begin(); itr != facetlist2.end(); ++itr){
	  QhullHyperplane plane = (*itr).hyperplane();
	  for (int i = 0; i < DIM; ++i)
		halfspaces.push_back(plane[i]);
	  halfspaces.push_back(plane.offset());
	}


    double interior_point[DIM] = {.5*(pcenter1[0]+pcenter2[0]),
										  .5*(pcenter1[1]+pcenter2[1]),
										  .5*(pcenter1[2]+pcenter2[2])};

    return qhull_volume_halfspace_intersection(
                                               (double *)&halfspaces[0],
                                               interior_point,
                                               halfspaces.size()/4,
                                               1.e10 // err_value
                                               );

    
  }

  catch(QhullError &e){
    // std::cout <<e.what() << std::endl;
  	return 1.e10;
  }


}


float diff_time(const std::chrono::time_point<std::chrono::high_resolution_clock> start){
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = stop-start;
  return std::chrono::duration<double>(stop-start).count();
}

// dist.shape = (n_polys, n_rays)
// points.shape = (n_polys, 3)
// verts.shape = (n_rays, 3)
// faces.shape = (n_faces, 3)
// expects that polys are sorted with associated descending scores
// returns boolean vector of polys indices that are kept

static PyObject* c_non_max_suppression_inds (PyObject *self, PyObject *args) {

  PyArrayObject *arr_dist=NULL, *arr_points=NULL,*arr_verts=NULL,*arr_faces=NULL,*arr_scores=NULL;
  PyArrayObject *arr_result=NULL;

  float threshold = 0;
  int use_bbox;
  int verbose;

  if (!PyArg_ParseTuple(args, "O!O!O!O!O!iif",
                        &PyArray_Type, &arr_dist,
                        &PyArray_Type, &arr_points,
                        &PyArray_Type, &arr_verts,
                        &PyArray_Type, &arr_faces,
                        &PyArray_Type, &arr_scores,
                        &use_bbox,
                        &verbose,
                        &threshold))
    return NULL;


  const int n_polys = PyArray_DIMS(arr_dist)[0];
  const int n_rays = PyArray_DIMS(arr_dist)[1];
  const int n_faces = PyArray_DIMS(arr_faces)[0];


  const float * const dist = (float*) PyArray_DATA(arr_dist);
  const float * const points = (float*) PyArray_DATA(arr_points);
  const float * const verts = (float*) PyArray_DATA(arr_verts);
  const int * const faces = (int*) PyArray_DATA(arr_faces);
  const float * const scores = (float*) PyArray_DATA(arr_scores);

  if (verbose){
    printf("Non Maximum Suppression (3D) ++++ \n");
    printf("NMS: n_polys  = %d \nNMS: n_rays   = %d  \nNMS: n_faces  = %d \nNMS: thresh   = %.3f \nNMS: use_bbox = %d \n", n_polys, n_rays, n_faces, threshold, use_bbox);
#ifdef _OPENMP
    printf("NMS: using OpenMP with %d thread(s)\n", omp_get_max_threads());
#endif
  }


  float * volumes = new float[n_polys];
  float * radius_inner = new float[n_polys];
  float * radius_outer = new float[n_polys];

  float * radius_inner_isotropic = new float[n_polys];
  float * radius_outer_isotropic = new float[n_polys];


  int * bbox = new int[6*n_polys];
  bool * suppressed = new bool[n_polys];
  float anisotropy[3] = {0.,0.,0};


  // first compute volumes, bounding boxes and anisotropy factors
  if (verbose)
    printf("NMS: precompute volumes, bounding boxes, etc\n");

#pragma omp parallel for
  for (int i=0; i<n_polys; i++) {

    const float * const curr_dist = &dist[i*n_rays];
    const float * const curr_point = &points[3*i];
    int * curr_bbox = &bbox[6*i];

    volumes[i] = polyhedron_volume(curr_dist, verts, faces, n_rays, n_faces);

    // // outer and inner bounding radius
    // radius_outer[i] = bounding_radius_outer(curr_dist, n_rays);
    // radius_inner[i] = bounding_radius_inner(curr_dist, verts, faces, n_rays, n_faces);

    // bounding boxes
    polyhedron_bbox(curr_dist, curr_point, verts, n_rays, curr_bbox);

    // running average of bbox anisotropies
    anisotropy[0] += (float)(curr_bbox[1]-curr_bbox[0])/n_polys;
    anisotropy[1] += (float)(curr_bbox[3]-curr_bbox[2])/n_polys;
    anisotropy[2] += (float)(curr_bbox[5]-curr_bbox[4])/n_polys;
  }

  // normalize and invert anisotropy (to resemble pixelsizes)
  // we would like to have anisotropy = (7,1,1) for highly axially anisotropic data
  float _tmp = fmax(fmax(anisotropy[0],anisotropy[1]),anisotropy[2]);
  anisotropy[0] = _tmp/anisotropy[0];
  anisotropy[1] = _tmp/anisotropy[1] ;
  anisotropy[2] = _tmp/anisotropy[2] ;

  if (verbose>=1)
    printf("NMS: calculated anisotropy: %.2f \t %.2f \t %.2f \n",anisotropy[0],anisotropy[1],anisotropy[2]);

  // calculate  bounding circles
#pragma omp parallel for
  for (int i=0; i<n_polys; i++) {

    const float * const curr_dist = &dist[i*n_rays];

    // outer and inner bounding radius
    radius_outer[i] = bounding_radius_outer(curr_dist, n_rays);
    radius_inner[i] = bounding_radius_inner(curr_dist, verts, faces, n_rays, n_faces);

    // outer and inner isotropic bounding radius
    radius_outer_isotropic[i] = bounding_radius_outer_isotropic(curr_dist, verts,
                                                                n_rays,anisotropy);

    radius_inner_isotropic[i] = bounding_radius_inner_isotropic(curr_dist, verts,
                                                                faces, n_rays,
                                                                n_faces, anisotropy);

    // printf("r    : %.2f \t %.2f \n",radius_inner[i], radius_outer[i]);
    // printf("r_iso: %.2f \t %.2f \n",radius_inner_isotropic[i], radius_outer_isotropic[i]);

  }



  // +++++++  NMS starts here ++++++++
  if (verbose)
    printf("NMS: starting suppression loop\n");

  int count_kept_pretest = 0;
  int count_kept_convex = 0;
  int count_suppressed_pretest = 0;
  int count_suppressed_kernel = 0;
  int count_suppressed_rendered = 0;

  int count_call_upper = 0;
  int count_call_lower = 0;
  int count_call_kernel = 0;
  int count_call_render = 0;
  int count_call_convex = 0;

  float timer_call_kernel = 0;
  float timer_call_convex = 0;
  float timer_call_render = 0;
   
  //initialize indices
  for (int i=0; i<n_polys; i++) {
    suppressed[i] = false;
  }

  float * curr_polyverts = new float[3*n_rays];

  ProgressBar prog("suppressed");
  
  // suppress (double loop)
  for (int i=0; i<n_polys-1; i++) {

    long count_total = count_suppressed_pretest+count_suppressed_kernel+count_suppressed_rendered;
    int status_percentage_new = 100*count_total/n_polys;

    // if verbose, print progress bar
    if (verbose){
      int count_total = count_suppressed_pretest+count_suppressed_kernel+count_suppressed_rendered;
      prog.update(100.*count_total/n_polys);
    }

    // check signals e.g. such that the loop is interruptable 
    if (PyErr_CheckSignals()==-1){
      delete [] volumes;
      delete [] curr_polyverts;
      delete [] bbox;
      delete [] suppressed;
      delete [] radius_inner;
      delete [] radius_outer;
      delete [] radius_inner_isotropic;
      delete [] radius_outer_isotropic;
      PyErr_SetString(PyExc_KeyboardInterrupt, "interupted");
      return Py_None;
    }

    // skip if already suppressed
    if (suppressed[i])
      continue;


    // the size of the bbox region of interest

    const float * const curr_dist  = &dist[i*n_rays];
    const float * const curr_point = &points[3*i];
    const int *   const curr_bbox  = &bbox[6*i];

    bool * curr_rendered = NULL;
     
    int Nz = curr_bbox[1]-curr_bbox[0]+1;
    int Ny = curr_bbox[3]-curr_bbox[2]+1;
    int Nx = curr_bbox[5]-curr_bbox[4]+1;

    // compute polyverts
    polyhedron_polyverts(curr_dist, curr_point, verts, n_rays, curr_polyverts);

     // printf("kernel volume step 1: %.2f\n", qhull_overlap_kernel(
     //                               curr_polyverts, curr_point,
     //                               curr_polyverts, curr_point,
     //                               faces, n_rays, n_faces, 1));
     // printf("kernel volume step 2: %.2f\n", qhull_overlap_kernel(
     //                               curr_polyverts, curr_point,
     //                               curr_polyverts, curr_point,
     //                               faces, n_rays, n_faces,2));
     // printf("kernel volume step 4: %.2f\n", qhull_overlap_kernel(
     //                               curr_polyverts, curr_point,
     //                               curr_polyverts, curr_point,
     //                               faces, n_rays, n_faces,4));
     
     // printf("kernel volume: %.2f\n", qhull_overlap_kernel(
     //                               curr_polyverts, curr_point,
     //                               curr_polyverts, curr_point,
	 //   								  faces, n_rays, n_faces));
     // printf("convex volume: %.2f\n", qhull_overlap_convex_hulls(
     //                               curr_polyverts, curr_point,
     //                               curr_polyverts, curr_point,
	 //   								  faces, n_rays, n_faces));

     
	 //  inner loop
	 //  can be parallelized....
#pragma omp parallel for schedule(dynamic) \
  reduction(+:count_suppressed_pretest) reduction(+:count_suppressed_kernel) \
  reduction(+:count_suppressed_rendered)                                \
  reduction(+:count_call_kernel) reduction(+:count_call_render)         \
  reduction(+:count_call_lower)                                         \
  reduction(+:count_call_upper) reduction(+:count_call_convex)          \
  reduction(+:count_kept_pretest) reduction(+:count_kept_convex)        \
  reduction(+:timer_call_kernel) reduction(+:timer_call_convex)        \
  reduction(+:timer_call_render) \
  shared(curr_rendered)

    for (int j=i+1; j<n_polys; j++) {

      if (suppressed[j])
        continue;

      std::chrono::time_point<std::chrono::high_resolution_clock> time_start;
      float iou = 0;
      float A_min = fmin(volumes[i], volumes[j]);
      float A_inter = 0;

      // --------- first check: bounding box and inner sphere intersection  (cheap)

             
      // upper  bound of intersection and IoU
      A_inter = fmin(intersect_sphere_isotropic(radius_outer_isotropic[i],
                                                &points[3*i],
                                                radius_outer_isotropic[j],
                                                &points[3*j],
                                                anisotropy
                                                ),
                     intersect_bbox(&bbox[6*i],&bbox[6*j]));
      count_call_upper++;

	   // if it doesn't intersect at all, we can move on...
	   iou = fmin(1.f,A_inter/(A_min+1e-10));

      if ((A_inter<1.e-10)||(iou<=threshold)){
        count_kept_pretest++;
        continue;
      }


      // lower bound of intersection and IoU
      A_inter = intersect_sphere_isotropic(radius_inner_isotropic[i],
                                           &points[3*i],
                                           radius_inner_isotropic[j],
                                           &points[3*j],
                                           anisotropy
                                           );
      count_call_lower++;

      // if lower bound is above threshold, we can safely suppress it...
      iou = fmax(0.f,A_inter/(A_min+1e-10));


      if (iou>threshold){
        count_suppressed_pretest++;
        suppressed[j] = true;
        continue;
      }

      float * polyverts = new float[3*n_rays];

      // compute polyverts of the second polyhedron
      polyhedron_polyverts(&dist[j*n_rays], &points[3*j],
                           verts,n_rays, polyverts);


      // ------- second check: kernel intersection (lower bound)

      time_start = std::chrono::high_resolution_clock::now();
       
      float A_inter_kernel = qhull_overlap_kernel(
	   								  curr_polyverts, curr_point,
	   								  polyverts, &points[3*j],
	   								  faces, n_rays, n_faces);

      count_call_kernel++;
      timer_call_kernel += diff_time(time_start);
       
      iou = A_inter_kernel/(A_min+1e-10);

       
      if (iou>threshold){
        count_suppressed_kernel++;
        suppressed[j] = true;
        delete[] polyverts;
        continue;
      }

      // ------- third check: intersection of convex hull (upper bound)
      time_start = std::chrono::high_resolution_clock::now();
      
      float A_inter_convex = qhull_overlap_convex_hulls(
	   								  curr_polyverts, curr_point,
	   								  polyverts, &points[3*j],
	   								  faces, n_rays, n_faces);
      count_call_convex++;
      timer_call_convex += diff_time(time_start);

      iou = A_inter_convex/(A_min+1e-10);

      if (iou<=threshold){
        count_kept_convex++;
        delete[] polyverts;
        continue;
      }


      // ------- forth/final check  (polygon rendering, exact)
       
      // render polyhedron
      time_start = std::chrono::high_resolution_clock::now();

      // check whether render buffer was already created 
      // if not, create it in a critical section  
#pragma omp critical
      if (!curr_rendered){
        {
          curr_rendered = new bool[Nz*Ny*Nx];
          render_polyhedron(curr_dist, curr_point, curr_bbox, curr_polyverts,
                            faces, n_rays, n_faces,  curr_rendered, Nz, Ny, Nx);
           
        }
      }
       
      float A_inter_render = overlap_render_polyhedron(&dist[j*n_rays],
                                                       &points[3*j],
                                                       curr_bbox,
                                                       polyverts,
                                                       faces, n_rays, n_faces,
                                                       curr_rendered, Nz, Ny, Nx,
                                                       (A_min+1e-10)*threshold
                                                       );
      count_call_render++;
      timer_call_render += diff_time(time_start);
      iou = A_inter_render/(A_min+1e-10);

      if (iou>threshold){
        count_suppressed_rendered++;
        suppressed[j] = true;
      }

      delete[] polyverts;
      
    }
    if (curr_rendered)
      delete [] curr_rendered;

  }

  if (verbose)
    prog.finish();
  
  if (verbose){
    printf("NMS: Function calls:\n");
    printf("NMS: ~ bbox+out: %8d\n", count_call_upper);
    printf("NMS: ~ inner:    %8d\n", count_call_lower);
    printf("NMS: ~ kernel:   %8d\n", count_call_kernel);
    printf("NMS: ~ convex:   %8d\n", count_call_convex);
    printf("NMS: ~ render:   %8d\n", count_call_render);

    printf("NMS: Excluded intersection:\n");
    printf("NMS: + pretest:  %8d\n", count_kept_pretest);
    printf("NMS: + convex:   %8d\n", count_kept_convex);
    printf("NMS: Function calls timing:\n");
    printf("NMS: / kernel:   %.2f s  (%.2f ms per call)\n", timer_call_kernel, 1000*timer_call_kernel/(1e-10+count_call_kernel));
    printf("NMS: / convex:   %.2f s  (%.2f ms per call)\n", timer_call_convex, 1000*timer_call_convex/(1e-10+count_call_convex));
    printf("NMS: / render:   %.2f s  (%.2f ms per call)\n", timer_call_render, 1000*timer_call_render/(1e-10+count_call_render));

    printf("NMS: Suppressed polyhedra:\n");
    printf("NMS: # inner:    %8d / %d  (%.2f %%)\n", count_suppressed_pretest,n_polys,100*(float)count_suppressed_pretest/n_polys);
    printf("NMS: # kernel:   %8d / %d  (%.2f %%)\n", count_suppressed_kernel,n_polys,100*(float)count_suppressed_kernel/n_polys);
    printf("NMS: # render:   %8d / %d  (%.2f %%)\n", count_suppressed_rendered,n_polys,100*(float)count_suppressed_rendered/n_polys);
    int count_suppressed_total = count_suppressed_pretest+count_suppressed_kernel+count_suppressed_rendered;
    printf("NMS: # total:    %8d / %d  (%.2f %%)\n", count_suppressed_total,n_polys,100*(float)count_suppressed_total/n_polys);
  }



  npy_intp dims_result[1];
  dims_result[0] = n_polys;
  arr_result = (PyArrayObject*)PyArray_SimpleNew(1,dims_result,NPY_BOOL);


  for (int i=0; i<n_polys;i++)
    *(bool *)PyArray_GETPTR1(arr_result,i) = !suppressed[i];

  delete [] volumes;
  delete [] curr_polyverts;
  delete [] bbox;
  delete [] suppressed;
  delete [] radius_inner;
  delete [] radius_outer;
  delete [] radius_inner_isotropic;
  delete [] radius_outer_isotropic;

  return PyArray_Return(arr_result);
}



//--------------------------------------------------------------

// dist.shape = (n_polys, n_rays)
// points.shape = (n_polys, 3)
// verts.shape = (n_rays, 3)
// faces.shape = (n_faces, 3)
// labels.shape = (n_polys,)
// shape = (3,)
// expects that polys are sorted with associated descending scores
// returns boolean vector of polys indices that are kept
//
// render_mode
// 0 -> full
// 1 -> kernel
// 2 -> bbox

static PyObject* c_polyhedron_to_label(PyObject *self, PyObject *args) {

  PyArrayObject *arr_dist=NULL, *arr_points=NULL,*arr_verts=NULL,*arr_faces=NULL,*arr_labels=NULL;
  PyArrayObject *arr_result=NULL;

  int nz,ny,nx;
  int render_mode;
  int verbose;
  int use_overlap_label;
  int overlap_label;
  
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiii(iii)",
                        &PyArray_Type, &arr_dist,
                        &PyArray_Type, &arr_points,
                        &PyArray_Type, &arr_verts,
                        &PyArray_Type, &arr_faces,
                        &PyArray_Type, &arr_labels,
                        &render_mode,
                        &verbose,
                        &use_overlap_label,
                        &overlap_label,
                        &nz,&ny,&nx))
    return NULL;


  const int n_polys = PyArray_DIMS(arr_dist)[0];
  const int n_rays = PyArray_DIMS(arr_dist)[1];
  const int n_faces = PyArray_DIMS(arr_faces)[0];


  const float * const dist = (float*) PyArray_DATA(arr_dist);
  const float * const points = (float*) PyArray_DATA(arr_points);
  const float * const verts = (float*) PyArray_DATA(arr_verts);
  const int * const faces = (int*) PyArray_DATA(arr_faces);
  const int * const labels = (int*) PyArray_DATA(arr_labels);


  npy_intp dims_result[3];
  dims_result[0] = nz;
  dims_result[1] = ny;
  dims_result[2] = nx;

  arr_result = (PyArrayObject*)PyArray_ZEROS(3,dims_result,NPY_INT32,0);

  if (verbose>=1){
    printf("+++++++++++++++ polyhedra to label +++++++++++++++ \n");
    printf("n_polys           = %d \n", n_polys);
    printf("n_rays            = %d \n", n_rays);
    printf("n_faces           = %d \n", n_faces);
    printf("nz, ny, nx        = %d %d %d \n", nz,ny,nx);
    printf("use_overlap_label = %d \n", use_overlap_label);
    printf("overlap_label     = %d \n", overlap_label);
  }

  float * polyverts = new float[3*n_rays];
  int bbox[6];


  for (int i = 0; i < n_polys; ++i) {

    const float * const curr_dist = &dist[i*n_rays];
    const float * const curr_center = &points[i*3];

    // calculate the actual vertices and the bounding box
    polyhedron_bbox(curr_dist, curr_center, verts, n_rays, bbox);
    polyhedron_polyverts(curr_dist, curr_center, verts, n_rays, polyverts);

    // get halfspaces of convex hull and of kernel
    std::vector<std::array<double,DIM+1>> hs_convex;
    std::vector<std::array<double,DIM+1>> hs_kernel;

    hs_convex = halfspaces_convex(polyverts, n_rays);
    hs_kernel = halfspaces_kernel(polyverts, faces, n_faces);

    // std::cout<<"halfspace convex "<<std::endl;
    // for(auto hs= hs_convex.begin();hs != hs_convex.end();hs++){
    //   for(int i = 0; i < DIM+1; ++i) {
    // 	 std::cout<< (*hs)[i] << " ";
    //   }
    //   std::cout<<std::endl;
    // }
    // std::cout<<"halfspace kernel "<<std::endl;
    // for(auto hs= hs_kernel.begin();hs != hs_kernel.end();hs++){
    //   for(int i = 0; i < DIM+1; ++i) {
    // 	 std::cout<< (*hs)[i] << " ";
    //   }
    //   std::cout<<std::endl;
    // }

    // loop over bounding box and label pixel if inside of the polyhedron
#pragma omp parallel for schedule(dynamic)
    for (int z = std::max(0,bbox[0]); z <= std::min(nz-1,bbox[1]); ++z) {
      for (int y = std::max(0,bbox[2]); y <= std::min(ny-1,bbox[3]); ++y) {
        for (int x = std::max(0,bbox[4]); x <= std::min(nx-1,bbox[5]); ++x) {

          // if ((*(int *)PyArray_GETPTR3(arr_result,z,y,x)==0) && (inside_polyhedron(z,y,x,center, polyverts, faces, n_rays, n_faces)))
          //   *(int *)PyArray_GETPTR3(arr_result,z,y,x) = labels[i];

          int * pval = (int *)PyArray_GETPTR3(arr_result,z,y,x);

          bool inside = false;

          switch(render_mode){
          case 0:
            // render_mode "full"
            // kernel and convex hull is fast, so we can formulate the condition
            // such that it can benefit from short-circuiting
            // inside  = in kernel OR (in convex hull AND in rendered)
            inside = (
                      point_in_halfspaces(z,y,x,hs_kernel) ||
                      (point_in_halfspaces(z,y,x,hs_convex) &&
                       inside_polyhedron(z,y,x, curr_center, polyverts, faces, n_rays, n_faces)));


            break;
          case 1:
            // render_mode "kernel"
            inside = point_in_halfspaces(z,y,x,hs_kernel);

            break;
          case 2:
            // render_mode "convex"
            inside = point_in_halfspaces(z,y,x,hs_convex);
            break;
          case 3:
            // render_mode "bbox"
            inside = true;
            break;
          case 4:
            // render_mode "debug"
            bool error = false;
            if ((inside_polyhedron_kernel(z,y,x,
                                          curr_center, polyverts,
                                          faces, n_rays, n_faces)) &&
                !(inside_polyhedron(z,y,x,
                                    curr_center, polyverts,
                                    faces, n_rays, n_faces)))
              error = true;

            if (error){
              *pval = -1;
              continue;
            }
            break;
          }

          if (inside){
            // if pixel is already labeled use this as the new label 
            int new_label_if_labeled = use_overlap_label?overlap_label:(*pval);

            *pval = (*pval)==0?labels[i]:new_label_if_labeled;
             
          }
        }
      }
    }
  }


  delete [] polyverts;

  return PyArray_Return(arr_result);
}

//------------------------------------------------------------------------


static PyObject* c_dist_to_volume(PyObject *self, PyObject *args) {

  PyArrayObject *arr_dist=NULL, *arr_verts=NULL,*arr_faces=NULL;
  PyArrayObject *arr_result=NULL;


  if (!PyArg_ParseTuple(args, "O!O!O!",
                        &PyArray_Type, &arr_dist,
                        &PyArray_Type, &arr_verts,
                        &PyArray_Type, &arr_faces))
    return NULL;


  const int nz = PyArray_DIMS(arr_dist)[0];
  const int ny = PyArray_DIMS(arr_dist)[1];
  const int nx = PyArray_DIMS(arr_dist)[2];
  const int n_rays = PyArray_DIMS(arr_dist)[3];
  const int n_faces = PyArray_DIMS(arr_faces)[0];


  const float * const dist = (float*) PyArray_DATA(arr_dist);
  const float * const verts = (float*) PyArray_DATA(arr_verts);
  const int * const faces = (int*) PyArray_DATA(arr_faces);


  const float origin[3] = {0.0, 0.0, 0.0};

  npy_intp dims_result[3];
  dims_result[0] = nz;
  dims_result[1] = ny;
  dims_result[2] = nx;

  arr_result = (PyArrayObject*)PyArray_ZEROS(3,dims_result,NPY_FLOAT32,0);

#pragma omp parallel for
  for (int k = 0; k < nz; ++k) {
	for (int j = 0; j < ny; ++j) {
	  for (int i = 0; i < nx; ++i) {

		float * polyverts = new float[3*n_rays];

		const int ind = n_rays*(i+nx*(j+k*ny));

		const float * curr_dist = &dist[ind];
		polyhedron_polyverts(curr_dist, origin, verts, n_rays, polyverts);

		const float vol = polyhedron_volume(&dist[ind], verts, faces, n_rays, n_faces);

		*(float *)PyArray_GETPTR3(arr_result,k,j,i) = vol;

		delete [] polyverts;
	  }
	}

  }

  return PyArray_Return(arr_result);
}

static PyObject* c_dist_to_centroid(PyObject *self, PyObject *args) {

  PyArrayObject *arr_dist=NULL, *arr_verts=NULL,*arr_faces=NULL;
  PyArrayObject *arr_result=NULL;

  int absolute;

  if (!PyArg_ParseTuple(args, "O!O!O!i",
                        &PyArray_Type, &arr_dist,
                        &PyArray_Type, &arr_verts,
						&PyArray_Type, &arr_faces,
						&absolute))
    return NULL;


  const int nz = PyArray_DIMS(arr_dist)[0];
  const int ny = PyArray_DIMS(arr_dist)[1];
  const int nx = PyArray_DIMS(arr_dist)[2];
  const int n_rays = PyArray_DIMS(arr_dist)[3];
  const int n_faces = PyArray_DIMS(arr_faces)[0];


  const float * const dist = (float*) PyArray_DATA(arr_dist);
  const float * const verts = (float*) PyArray_DATA(arr_verts);
  const int * const faces = (int*) PyArray_DATA(arr_faces);


  const float origin[3] = {0.0, 0.0, 0.0};


  npy_intp dims_result[4];
  dims_result[0] = nz;
  dims_result[1] = ny;
  dims_result[2] = nx;
  dims_result[3] = 3;

  arr_result = (PyArrayObject*)PyArray_ZEROS(4,dims_result,NPY_FLOAT32,0);

#pragma omp parallel for
  for (int k = 0; k < nz; ++k) {
	for (int j = 0; j < ny; ++j) {
	  for (int i = 0; i < nx; ++i) {
		const int ind = n_rays*(i+nx*(j+k*ny));

		const float * curr_dist = &dist[ind];
		float * polyverts = new float[3*n_rays];
		float centroid[3];

		polyhedron_polyverts(curr_dist, origin, verts, n_rays, polyverts);

	    polyhedron_centroid(&dist[ind], verts, faces, n_rays, n_faces, centroid);


		*(float *)PyArray_GETPTR4(arr_result,k,j,i,0) = centroid[0] + absolute*k;
		*(float *)PyArray_GETPTR4(arr_result,k,j,i,1) = centroid[1] + absolute*j;
		*(float *)PyArray_GETPTR4(arr_result,k,j,i,2) = centroid[2] + absolute*i;
		delete [] polyverts;
	  }
	}

  }


  return PyArray_Return(arr_result);
}

// -----------------------------------------------------------------------


// create the star convex distances from a labeled image

static PyObject* c_star_dist3d(PyObject *self, PyObject *args) {

  PyArrayObject *src = NULL;
  PyArrayObject *dst = NULL;

  PyArrayObject *pdx = NULL;
  PyArrayObject *pdy = NULL;
  PyArrayObject *pdz = NULL;


  int n_rays;
  int grid_x, grid_y, grid_z;


  if (!PyArg_ParseTuple(args, "O!O!O!O!iiii", &PyArray_Type, &src, &PyArray_Type, &pdz ,&PyArray_Type, &pdy,&PyArray_Type, &pdx, &n_rays,&grid_z,&grid_y,&grid_x))
    return NULL;

  npy_intp *dims = PyArray_DIMS(src);

  npy_intp dims_dst[4];
  dims_dst[0] = dims[0]/grid_z;
  dims_dst[1] = dims[1]/grid_y;
  dims_dst[2] = dims[2]/grid_x;
  dims_dst[3] = n_rays;

  dst = (PyArrayObject*)PyArray_SimpleNew(4,dims_dst,NPY_FLOAT32);

# pragma omp parallel for schedule(dynamic)
  for (int i=0; i<dims_dst[0]; i++) {
    for (int j=0; j<dims_dst[1]; j++) {
      for (int k=0; k<dims_dst[2]; k++) {
        const unsigned short value = *(unsigned short *)PyArray_GETPTR3(src,i*grid_z,j*grid_y,k*grid_x);
        // background pixel
        if (value == 0) {
          for (int n = 0; n < n_rays; n++) {
            *(float *)PyArray_GETPTR4(dst,i,j,k,n) = 0;
          }
          // foreground pixel
        } else {

          for (int n = 0; n < n_rays; n++) {

            float dx = *(float *)PyArray_GETPTR1(pdx,n);
            float dy = *(float *)PyArray_GETPTR1(pdy,n);
            float dz = *(float *)PyArray_GETPTR1(pdz,n);

            float x = 0, y = 0, z=0;
            // move along ray
            while (1) {
              x += dx;
              y += dy;
              z += dz;
              const int ii = round_to_int(i*grid_z+z), jj = round_to_int(j*grid_y+y), kk = round_to_int(k*grid_x+x);

              //std::cout<<"ii: "<<ii<<" vs  "<<i*grid_z+z<<std::endl;

              // stop if out of bounds or reaching a pixel with a different value/id
              if (ii < 0 || ii >= dims[0] ||
                  jj < 0 || jj >= dims[1] ||
                  kk < 0 || kk >= dims[2] ||
                  value != *(unsigned short *)PyArray_GETPTR3(src,ii,jj,kk))
                {
                  const float dist = sqrt(x*x + y*y + z*z);

                  // const int x2 = round_to_int(x);
                  // const int y2 = round_to_int(y);
                  // const int z2 = round_to_int(z);
                  // const float dist = sqrt(x2*x2 + y2*y2 + z2*z2);

                  *(float *)PyArray_GETPTR4(dst,i,j,k,n) = dist;
                  break;
                }
            }
          }
        }

      }
    }
  }

  return PyArray_Return(dst);
}

//------------------------------------------------------------------------

static struct PyMethodDef methods[] = {
                                       {"c_star_dist3d",
                                        c_star_dist3d,
                                        METH_VARARGS,
                                        "star dist 3d calculation"},
                                       
                                       {"c_non_max_suppression_inds",
                                        c_non_max_suppression_inds,
                                        METH_VARARGS,
                                        "non-maximum suppression"},
                                       
                                       {"c_polyhedron_to_label",
                                        c_polyhedron_to_label,
                                        METH_VARARGS,
                                        "polyhedron to label"},
                                       
                                       {"c_dist_to_volume",
                                        c_dist_to_volume,
                                        METH_VARARGS,
                                        "distance to volume"},
                                       
                                       {"c_dist_to_centroid",
                                        c_dist_to_centroid,
                                        METH_VARARGS,
                                        "distance to centroids"},
                                       {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
                                       PyModuleDef_HEAD_INIT,
                                       "stardist3d", 
                                       NULL,         
                                       -1,           
                                       methods,
                                       NULL,NULL,NULL,NULL
};

PyMODINIT_FUNC PyInit_stardist3d(void) {
  import_array();
  return PyModule_Create(&moduledef);
}