#ifndef LIBSTARDIST3D_H
#define LIBSTARDIST3D_H

void _callback_non_maximum_suppression_sparse(
                    const float* scores, const float* dist, const float* points,
                    const int n_polys, const int n_rays, const int n_faces, 
                    const float* verts, const int* faces,
                    bool* result);


void _callback_polyhedron_to_label(const float* dist, const float* points,
                                   const float* verts,const int* faces,
                                   const int n_polys, const int n_rays, const int n_faces,
                                   const int* labels,
                                   const int nz, const int  ny,const int nx,
                                   int * result);

#ifdef __cplusplus
extern "C" {
#endif
 
void non_maximum_suppression_sparse(
                    const float*, const float*, const float*,                    
                    const int, const int, const int,
                    const float* , const int* ,
                    bool*);

void  polyhedron_to_label(const float* dist, const float* points,
                          const float* verts,const int* faces,
                          const int n_polys, const int n_rays, const int n_faces,
                          const int* labels,
                          const int nz, const int  ny,const int nx,
                          int * result);
  
#ifdef __cplusplus
}
#endif


#endif /* LIBSTARDIST3D_H */
