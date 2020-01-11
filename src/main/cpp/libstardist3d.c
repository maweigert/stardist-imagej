#include "libstardist3d.h"
 
void non_maximum_suppression_sparse(
                    const float* scores, const float* dist, const float* points,
                    const int n_polys, const int n_rays, const int n_faces,
                    const float* verts, const int* faces,
                    bool* result)
{
  _callback_non_maximum_suppression_sparse(scores, dist, points,
                                           n_polys, n_rays, n_faces,
                                           verts, faces,
                                           result );
}


void polyhedron_to_label(const float* dist, const float* points,
                                    const float* verts,const int* faces,
                                    const int n_polys, const int n_rays, const int n_faces,                                    
                                    const int* labels,
                         const int nz, const int  ny,const int nx,
                                                            int * result)
{
  _callback_polyhedron_to_label(dist, points,
                                verts,faces,
                                n_polys,n_rays, n_faces, 
                                labels,
                                nz, ny,nx,
                                result);

}
