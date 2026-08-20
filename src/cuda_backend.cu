#include "cuda_backend.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace kami {
namespace {

#define CUDA_TRY(call) do { const cudaError_t khs_cuda_error = (call); \
    if (khs_cuda_error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(khs_cuda_error)); } while (0)
#define CUBLAS_TRY(call) do { const cublasStatus_t khs_cublas_error = (call); \
    if (khs_cublas_error != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error"); } while (0)

constexpr double kPi = 3.1415926535897932384626433832795;
constexpr int kThreads = 128;
constexpr int kBvhStack = 64;
constexpr uint64_t kCheckpointMagic = 0x3154504348534bULL;

struct DVec3 {
    double x, y, z;
};

__host__ __device__ DVec3 make_vec(double x = 0.0, double y = 0.0, double z = 0.0)
{
    return {x, y, z};
}

__host__ __device__ DVec3 operator+(DVec3 a, DVec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
__host__ __device__ DVec3 operator-(DVec3 a, DVec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
__host__ __device__ DVec3 operator*(DVec3 a, double s) { return {a.x*s, a.y*s, a.z*s}; }
__host__ __device__ DVec3 operator*(double s, DVec3 a) { return a*s; }
__host__ __device__ DVec3 operator/(DVec3 a, double s) { return a*(1.0/s); }
__host__ __device__ double dot(DVec3 a, DVec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__host__ __device__ DVec3 cross(DVec3 a, DVec3 b)
{
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
__host__ __device__ double norm2(DVec3 a) { return dot(a,a); }
__host__ __device__ double norm(DVec3 a) { return sqrt(norm2(a)); }
__host__ __device__ DVec3 min_vec(DVec3 a, DVec3 b)
{
    return {fmin(a.x,b.x),fmin(a.y,b.y),fmin(a.z,b.z)};
}
__host__ __device__ DVec3 max_vec(DVec3 a, DVec3 b)
{
    return {fmax(a.x,b.x),fmax(a.y,b.y),fmax(a.z,b.z)};
}

template<int N> struct Dual {
    double v;
    double d[N];
    __device__ Dual(double value = 0.0) : v(value) { for (int i=0;i<N;++i) d[i]=0.0; }
    __device__ static Dual variable(double value, int index) {
        Dual out(value); out.d[index]=1.0; return out;
    }
};

template<int N> __device__ Dual<N> operator+(const Dual<N>&a,const Dual<N>&b) {
    Dual<N> r(a.v+b.v); for(int i=0;i<N;++i) r.d[i]=a.d[i]+b.d[i]; return r;
}
template<int N> __device__ Dual<N> operator-(const Dual<N>&a,const Dual<N>&b) {
    Dual<N> r(a.v-b.v); for(int i=0;i<N;++i) r.d[i]=a.d[i]-b.d[i]; return r;
}
template<int N> __device__ Dual<N> operator-(const Dual<N>&a) {
    Dual<N> r(-a.v); for(int i=0;i<N;++i) r.d[i]=-a.d[i]; return r;
}
template<int N> __device__ Dual<N> operator*(const Dual<N>&a,const Dual<N>&b) {
    Dual<N> r(a.v*b.v); for(int i=0;i<N;++i) r.d[i]=a.d[i]*b.v+a.v*b.d[i]; return r;
}
template<int N> __device__ Dual<N> operator/(const Dual<N>&a,const Dual<N>&b) {
    const double inv=1.0/b.v; Dual<N> r(a.v*inv);
    for(int i=0;i<N;++i) r.d[i]=(a.d[i]-r.v*b.d[i])*inv; return r;
}
template<int N> __device__ Dual<N> dsqrt(const Dual<N>&a) {
    const double s=sqrt(a.v); Dual<N> r(s); const double f=0.5/fmax(s,1.0e-300);
    for(int i=0;i<N;++i) r.d[i]=a.d[i]*f; return r;
}
template<int N> __device__ Dual<N> dsin(const Dual<N>&a) {
    Dual<N> r(sin(a.v)); const double c=cos(a.v); for(int i=0;i<N;++i) r.d[i]=c*a.d[i]; return r;
}
template<int N> __device__ Dual<N> dcos(const Dual<N>&a) {
    Dual<N> r(cos(a.v)); const double s=-sin(a.v); for(int i=0;i<N;++i) r.d[i]=s*a.d[i]; return r;
}
template<int N> __device__ Dual<N> dlog(const Dual<N>&a) {
    Dual<N> r(log(a.v)); const double inv=1.0/a.v; for(int i=0;i<N;++i) r.d[i]=inv*a.d[i]; return r;
}

template<typename S> struct TVec3 { S x,y,z; };
template<typename S> __device__ TVec3<S> tvadd(TVec3<S>a,TVec3<S>b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
template<typename S> __device__ TVec3<S> tvsub(TVec3<S>a,TVec3<S>b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
template<typename S> __device__ TVec3<S> tvscale(TVec3<S>a,S s){return {a.x*s,a.y*s,a.z*s};}
template<typename S> __device__ S tvdot(TVec3<S>a,TVec3<S>b){return a.x*b.x+a.y*b.y+a.z*b.z;}

template<typename S> struct TMat3 { S v[9]; };
template<typename S> __device__ TMat3<S> mat_identity() {
    TMat3<S> r{}; r.v[0]=S(1);r.v[4]=S(1);r.v[8]=S(1);return r;
}
template<typename S> __device__ TMat3<S> mat_add(const TMat3<S>&a,const TMat3<S>&b) {
    TMat3<S>r;for(int i=0;i<9;++i)r.v[i]=a.v[i]+b.v[i];return r;
}
template<typename S> __device__ TMat3<S> mat_scale(const TMat3<S>&a,S s) {
    TMat3<S>r;for(int i=0;i<9;++i)r.v[i]=a.v[i]*s;return r;
}
template<typename S> __device__ TMat3<S> mat_mul(const TMat3<S>&a,const TMat3<S>&b) {
    TMat3<S>r{};for(int y=0;y<3;++y)for(int x=0;x<3;++x)for(int k=0;k<3;++k)
        r.v[3*y+x]=r.v[3*y+x]+a.v[3*y+k]*b.v[3*k+x];return r;
}
template<typename S> __device__ TMat3<S> mat_transpose(const TMat3<S>&a) {
    TMat3<S>r;for(int y=0;y<3;++y)for(int x=0;x<3;++x)r.v[3*y+x]=a.v[3*x+y];return r;
}
template<typename S> __device__ TVec3<S> mat_vec(const TMat3<S>&a,TVec3<S>b) {
    return {a.v[0]*b.x+a.v[1]*b.y+a.v[2]*b.z,
            a.v[3]*b.x+a.v[4]*b.y+a.v[5]*b.z,
            a.v[6]*b.x+a.v[7]*b.y+a.v[8]*b.z};
}
template<typename S> __device__ TMat3<S> skew(TVec3<S>w) {
    TMat3<S>r{};r.v[1]=-w.z;r.v[2]=w.y;r.v[3]=w.z;r.v[5]=-w.x;r.v[6]=-w.y;r.v[7]=w.x;return r;
}

template<int N> __device__ TMat3<Dual<N>> exp_so3(TVec3<Dual<N>> w) {
    const Dual<N> t2=tvdot(w,w); Dual<N>a,b;
    if(t2.v<1.0e-10){a=Dual<N>(1)-t2/Dual<N>(6)+t2*t2/Dual<N>(120);
        b=Dual<N>(0.5)-t2/Dual<N>(24)+t2*t2/Dual<N>(720);}
    else {const Dual<N>t=dsqrt(t2);a=dsin(t)/t;b=(Dual<N>(1)-dcos(t))/t2;}
    const auto W=skew(w);return mat_add(mat_identity<Dual<N>>(),mat_add(mat_scale(W,a),mat_scale(mat_mul(W,W),b)));
}

template<int N> __device__ TVec3<Dual<N>> cayley(const TMat3<Dual<N>>&q) {
    TVec3<Dual<N>>v{(q.v[7]-q.v[5])*Dual<N>(0.5),(q.v[2]-q.v[6])*Dual<N>(0.5),
        (q.v[3]-q.v[1])*Dual<N>(0.5)};
    const Dual<N>c=(q.v[0]+q.v[4]+q.v[8]-Dual<N>(1))*Dual<N>(0.5);
    return tvscale(v,Dual<N>(2)/(Dual<N>(1)+c+Dual<N>(1.0e-12)));
}

struct DNode {
    DVec3 reference[3];
    double mass;
    double rotational_mass;
    uint32_t strand;
    uint32_t binding_left;
    uint32_t binding_right;
    double binding_t;
    uint32_t fixed;
};
struct DElement { uint32_t i,j,strand,collider_contact; double rest_length; DVec3 rest_shear,rest_curvature; };
struct DStrand { uint32_t first,count,fixed_count; };
struct DTriangle { uint32_t i0,i1,i2; };
struct DBvhNode { DVec3 lower,upper; uint32_t begin,count; int left,right; };
struct DMaterial {
    double density,radius,young_modulus,poisson_ratio,shear_correction,mass_damping;
    double contact_stiffness,barrier_distance,friction,friction_smoothing,collider_offset;
};
struct DDesc {
    DVec3 gravity;
    uint32_t substeps,maximum_substeps,newton_iterations,line_search_iterations;
    double absolute_tolerance,relative_tolerance,increment_tolerance;
    double minimum_line_search_step,minimum_gap;
};
struct DAssemblyStats {
    double objective,elastic,contact,friction,kinetic,minimum_gap;
    unsigned long long candidates,active;
    int feasible;
};

struct DMovingSweepFailure {
    int claimed;
    uint32_t strand_index,element_index,triangle_index;
    double distance,required_distance,clearance,collider_displacement;
    DVec3 hair_start,hair_end,collider_point;
};

struct AnimationCheckpointHeader {
    uint64_t magic;
    uint32_t abi_version;
    uint32_t animation_frame;
    uint64_t dof_count;
};

__device__ void atomic_min_double(double *address,double value) {
    auto *p=reinterpret_cast<unsigned long long*>(address); unsigned long long old=*p,assumed;
    while(__longlong_as_double(old)>value){assumed=old;old=atomicCAS(p,assumed,__double_as_longlong(value));if(old==assumed)break;}
}

__device__ void atomic_max_double(double *address,double value) {
    auto *p=reinterpret_cast<unsigned long long*>(address); unsigned long long old=*p,assumed;
    while(__longlong_as_double(old)<value){assumed=old;old=atomicCAS(p,assumed,__double_as_longlong(value));if(old==assumed)break;}
}

struct ClosestPair { double distance,t; DVec3 bary; };
__device__ DVec3 closest_point_triangle(DVec3 p,DVec3 a,DVec3 b,DVec3 c,DVec3&bary) {
    const DVec3 ab=b-a,ac=c-a,ap=p-a;const double d1=dot(ab,ap),d2=dot(ac,ap);
    if(d1<=0&&d2<=0){bary=make_vec(1,0,0);return a;}const DVec3 bp=p-b;
    const double d3=dot(ab,bp),d4=dot(ac,bp);if(d3>=0&&d4<=d3){bary=make_vec(0,1,0);return b;}
    const double vc=d1*d4-d3*d2;if(vc<=0&&d1>=0&&d3<=0){double v=d1/(d1-d3);bary=make_vec(1-v,v,0);return a+ab*v;}
    const DVec3 cp=p-c;const double d5=dot(ab,cp),d6=dot(ac,cp);if(d6>=0&&d5<=d6){bary=make_vec(0,0,1);return c;}
    const double vb=d5*d2-d1*d6;if(vb<=0&&d2>=0&&d6<=0){double w=d2/(d2-d6);bary=make_vec(1-w,0,w);return a+ac*w;}
    const double va=d3*d6-d5*d4;if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0){double w=(d4-d3)/((d4-d3)+(d5-d6));bary=make_vec(0,1-w,w);return b+(c-b)*w;}
    const double inv=1.0/(va+vb+vc),v=vb*inv,w=vc*inv;bary=make_vec(1-v-w,v,w);return a+ab*v+ac*w;
}
__device__ void consider_segment(DVec3 p1,DVec3 q1,DVec3 p2,DVec3 q2,double&best,double&bs,double&bt) {
    const DVec3 d1=q1-p1,d2=q2-p2,r=p1-p2;const double a=dot(d1,d1),e=dot(d2,d2),f=dot(d2,r);double s=0,t=0;
    if(a<=1e-14&&e<=1e-14){}else if(a<=1e-14)t=fmin(1.0,fmax(0.0,f/e));else{const double c=dot(d1,r);
        if(e<=1e-14)s=fmin(1.0,fmax(0.0,-c/a));else{const double b=dot(d1,d2),den=a*e-b*b;if(den!=0)s=fmin(1.0,fmax(0.0,(b*f-c*e)/den));t=(b*s+f)/e;
            if(t<0){t=0;s=fmin(1.0,fmax(0.0,-c/a));}else if(t>1){t=1;s=fmin(1.0,fmax(0.0,(b-c)/a));}}}
    const double d=norm((p1+d1*s)-(p2+d2*t));if(d<best){best=d;bs=s;bt=t;}
}
__device__ ClosestPair closest_segment_triangle(DVec3 p0,DVec3 p1,DVec3 a,DVec3 b,DVec3 c) {
    ClosestPair out{1.0e300,0,make_vec(1,0,0)};DVec3 bary,cp=closest_point_triangle(p0,a,b,c,bary);double d=norm(p0-cp);
    if(d<out.distance)out={d,0,bary};cp=closest_point_triangle(p1,a,b,c,bary);d=norm(p1-cp);if(d<out.distance)out={d,1,bary};
    DVec3 tv[3]{a,b,c};for(int e=0;e<3;++e){double best=out.distance,s=0,t=0;consider_segment(p0,p1,tv[e],tv[(e+1)%3],best,s,t);
        if(best<out.distance){out.distance=best;out.t=s;out.bary=make_vec();if(e==0){out.bary.x=1-t;out.bary.y=t;}else if(e==1){out.bary.y=1-t;out.bary.z=t;}else{out.bary.z=1-t;out.bary.x=t;}}}
    return out;
}

__device__ bool overlap(DVec3 al,DVec3 au,DVec3 bl,DVec3 bu,double p) {
    return al.x<=bu.x+p&&bl.x<=au.x+p&&al.y<=bu.y+p&&bl.y<=au.y+p&&al.z<=bu.z+p&&bl.z<=au.z+p;
}

template<class F> __device__ void query_bvh(const DBvhNode*nodes,const uint32_t*order,DVec3 lo,DVec3 hi,double padding,F&&f) {
    int stack[kBvhStack],top=0;stack[top++]=0;while(top){const DBvhNode&n=nodes[stack[--top]];if(!overlap(lo,hi,n.lower,n.upper,padding))continue;
        if(n.left>=0){if(top+2<=kBvhStack){stack[top++]=n.left;stack[top++]=n.right;}}else for(uint32_t k=n.begin;k<n.begin+n.count;++k)f(order[k]);}
}

template<int N> __device__ void rod_strains(const Dual<N>*q,const DNode&ni,const DNode&nj,
                                           const DElement&e,Dual<N>*strain) {
    TVec3<Dual<N>>xi{q[0],q[1],q[2]},ui{q[3],q[4],q[5]},xj{q[6],q[7],q[8]},uj{q[9],q[10],q[11]};
    TMat3<Dual<N>>refi,refj;for(int k=0;k<9;++k){refi.v[k]=Dual<N>((&ni.reference[0].x)[k]);refj.v[k]=Dual<N>((&nj.reference[0].x)[k]);}
    const auto ri=mat_mul(exp_so3(ui),refi),rj=mat_mul(exp_so3(uj),refj);
    const auto tangent=tvscale(tvsub(xj,xi),Dual<N>(1.0/e.rest_length));
    const auto shear=tvscale(mat_vec(mat_add(mat_transpose(ri),mat_transpose(rj)),tangent),Dual<N>(0.5));
    const auto curvature=tvscale(cayley(mat_mul(mat_transpose(ri),rj)),Dual<N>(1.0/e.rest_length));
    const auto ds=tvsub(shear,TVec3<Dual<N>>{Dual<N>(e.rest_shear.x),Dual<N>(e.rest_shear.y),Dual<N>(e.rest_shear.z)});
    const auto dk=tvsub(curvature,TVec3<Dual<N>>{Dual<N>(e.rest_curvature.x),Dual<N>(e.rest_curvature.y),Dual<N>(e.rest_curvature.z)});
    strain[0]=ds.x;strain[1]=ds.y;strain[2]=ds.z;strain[3]=dk.x;strain[4]=dk.y;strain[5]=dk.z;
}

template<int N> __device__ Dual<N> rod_energy(const Dual<N>*q,const DNode&ni,const DNode&nj,const DElement&e,const DMaterial&m) {
    Dual<N> strain[6];rod_strains(q,ni,nj,e,strain);
    const double area=kPi*m.radius*m.radius,inertia=kPi*m.radius*m.radius*m.radius*m.radius/4.0,polar=2*inertia;
    const double shear_modulus=m.young_modulus/(2*(1+m.poisson_ratio));
    return Dual<N>(0.5*e.rest_length)*(Dual<N>(m.shear_correction*shear_modulus*area)*(strain[0]*strain[0]+strain[1]*strain[1])+
        Dual<N>(m.young_modulus*area)*strain[2]*strain[2]+Dual<N>(m.young_modulus*inertia)*(strain[3]*strain[3]+strain[4]*strain[4])+Dual<N>(shear_modulus*polar)*strain[5]*strain[5]);
}

__device__ double effective_contact_stiffness(const DNode*nodes,const DElement&e,double t,const DMaterial&m,double gap,double dt) {
    const double w0=1.0-t,w1=t;double inverse_mass=0.0;
    if(!nodes[e.i].fixed)inverse_mass+=w0*w0/fmax(nodes[e.i].mass,1.0e-30);
    if(!nodes[e.j].fixed)inverse_mass+=w1*w1/fmax(nodes[e.j].mass,1.0e-30);
    if(!(inverse_mass>0.0))return m.contact_stiffness;
    // Match the projected PPF barrier scale: inertia keeps contact effective at
    // short time steps, while mass/gap^2 prevents TOI slices from ratcheting to zero.
    const double effective_mass=1.0/inverse_mass;
    const double projected_stiffness=effective_mass*(1.0/fmax(dt*dt,1.0e-30)+1.0/fmax(gap*gap,1.0e-30));
    return fmax(m.contact_stiffness,projected_stiffness);
}

template<int N> __device__ Dual<N> contact_energy_dual(const Dual<N>*q,DVec3 old0,DVec3 old1,DVec3 a,DVec3 b,DVec3 c,DVec3 oa,DVec3 ob,DVec3 oc,const ClosestPair&pair,const DMaterial&m,double contact_stiffness) {
    (void)old0;(void)old1;(void)oa;(void)ob;(void)oc;
    TVec3<Dual<N>>x0{q[0],q[1],q[2]},x1{q[3],q[4],q[5]};const Dual<N>t(pair.t);
    const auto p=tvadd(tvscale(x0,Dual<N>(1)-t),tvscale(x1,t));
    const DVec3 bp=a*pair.bary.x+b*pair.bary.y+c*pair.bary.z;
    const auto diff=tvsub(p,TVec3<Dual<N>>{Dual<N>(bp.x),Dual<N>(bp.y),Dual<N>(bp.z)});const Dual<N>dist=dsqrt(tvdot(diff,diff)+Dual<N>(1e-30));
    const Dual<N>gap=dist-Dual<N>(m.radius+m.collider_offset);if(gap.v>=m.barrier_distance)return q[0]*Dual<N>(0);
    const Dual<N>aa=Dual<N>(m.barrier_distance)-gap,lg=dlog(gap/Dual<N>(m.barrier_distance));
    const Dual<N>barrier=-Dual<N>(contact_stiffness)*aa*aa*lg;
    return barrier;
}

} // namespace
} // namespace kami

namespace kami {
namespace {

__global__ void reset_stats_kernel(DAssemblyStats*);
__global__ void interpolate_collider_kernel(DVec3*,const DVec3*,const DVec3*,uint32_t,double);
__global__ void refit_leaves_kernel(DBvhNode*,const uint32_t*,uint32_t,const uint32_t*,const DTriangle*,const DVec3*,const DVec3*,bool);
__global__ void refit_level_kernel(DBvhNode*,const uint32_t*,uint32_t);
__global__ void begin_substep_kernel(double*,const double*,double*,const DNode*,uint32_t,const DVec3*,const DVec3*,const DVec3*,const DVec3*,double,double);
__global__ void update_velocity_kernel(const double*,const double*,double*,const DNode*,uint32_t,double);
__global__ void enforce_fixed_direction_kernel(double*,const DNode*,uint32_t);
__global__ void preconditioner_node_kernel(double*,const DNode*,uint32_t,DMaterial,double);
__global__ void preconditioner_element_kernel(double*,const DElement*,uint32_t,DMaterial);
__global__ void preconditioner_finish_kernel(double*,const DNode*,uint32_t);
__global__ void multiply_kernel(double*,const double*,uint32_t);
__global__ void maximum_translation_kernel(const double*,const DNode*,uint32_t,double*);
__global__ void node_energy_kernel(const double*,const double*,const double*,const DNode*,uint32_t,DDesc,DMaterial,double,double*,double*,DAssemblyStats*);
__global__ void element_energy_kernel(const double*,const double*,const DNode*,const DElement*,uint32_t,DMaterial,DDesc,double,const DVec3*,const DVec3*,const DTriangle*,const DBvhNode*,const uint32_t*,bool,double*,double*,double*,DAssemblyStats*);
__global__ void block_solve_kernel(const DStrand*,uint32_t,const double*,const double*,const double*,double*,double*,double*,int*);
__global__ void moving_sweep_kernel(const double*,const DNode*,const DElement*,uint32_t,const DVec3*,const DVec3*,const DTriangle*,const DBvhNode*,const uint32_t*,DMaterial,DDesc,double*,DMovingSweepFailure*);
__global__ void ccd_limit_kernel(const double*,const double*,const DNode*,const DElement*,uint32_t,const DVec3*,const DTriangle*,const DBvhNode*,const uint32_t*,DMaterial,DDesc,double*);
__global__ void friction_impulse_kernel(const double*,double*,const double*,const DNode*,const DElement*,uint32_t,const DVec3*,const DVec3*,const DTriangle*,const DBvhNode*,const uint32_t*,DMaterial,double);
__global__ void gather_original_kernel(const double*,const uint32_t*,uint32_t,DVec3*);
__global__ void gather_internal_kernel(const double*,uint32_t,DVec3*);

struct HostBvh {
    std::vector<DBvhNode> nodes;
    std::vector<uint32_t> order;
    std::vector<std::vector<uint32_t>> levels;
    std::vector<uint32_t> leaves;
    const std::vector<DVec3>*vertices=nullptr;const std::vector<DTriangle>*triangles=nullptr;
    int build_node(uint32_t begin,uint32_t end,uint32_t depth) {
        const int idx=static_cast<int>(nodes.size());nodes.push_back({});if(levels.size()<=depth)levels.resize(depth+1);levels[depth].push_back(idx);
        DVec3 lo=make_vec(1e300,1e300,1e300),hi=make_vec(-1e300,-1e300,-1e300),cl=lo,cu=hi;
        for(uint32_t k=begin;k<end;++k){const auto&t=(*triangles)[order[k]];const DVec3 a=(*vertices)[t.i0],b=(*vertices)[t.i1],c=(*vertices)[t.i2];lo=min_vec(lo,min_vec(a,min_vec(b,c)));hi=max_vec(hi,max_vec(a,max_vec(b,c)));const DVec3 ce=(a+b+c)/3.0;cl=min_vec(cl,ce);cu=max_vec(cu,ce);}
        nodes[idx].lower=lo;nodes[idx].upper=hi;nodes[idx].begin=begin;nodes[idx].count=end-begin;nodes[idx].left=nodes[idx].right=-1;
        if(end-begin<=8){leaves.push_back(idx);return idx;}const DVec3 span=cu-cl;const int axis=span.y>span.x?(span.z>span.y?2:1):(span.z>span.x?2:0);const uint32_t mid=begin+(end-begin)/2;
        std::nth_element(order.begin()+begin,order.begin()+mid,order.begin()+end,[&](uint32_t l,uint32_t r){const auto&tl=(*triangles)[l];const auto&tr=(*triangles)[r];
            const DVec3 lc=((*vertices)[tl.i0]+(*vertices)[tl.i1]+(*vertices)[tl.i2])/3.0,rc=((*vertices)[tr.i0]+(*vertices)[tr.i1]+(*vertices)[tr.i2])/3.0;return (&lc.x)[axis]<(&rc.x)[axis];});
        const int left=build_node(begin,mid,depth+1),right=build_node(mid,end,depth+1);nodes[idx].left=left;nodes[idx].right=right;nodes[idx].count=0;return idx;
    }
    void build(const std::vector<DVec3>&v,const std::vector<DTriangle>&t){vertices=&v;triangles=&t;order.resize(t.size());std::iota(order.begin(),order.end(),0);if(!t.empty())build_node(0,uint32_t(t.size()),0);}
};

template<typename T> class DeviceBuffer {
public:
    DeviceBuffer()=default;~DeviceBuffer(){reset();}DeviceBuffer(const DeviceBuffer&)=delete;DeviceBuffer&operator=(const DeviceBuffer&)=delete;
    void allocate(size_t count){reset();count_=count;if(count)CUDA_TRY(cudaMalloc(&ptr_,count*sizeof(T)));}
    void reset(){if(ptr_)cudaFree(ptr_);ptr_=nullptr;count_=0;}
    T*get(){return ptr_;}T*get()const{return ptr_;}size_t size()const{return count_;}size_t bytes()const{return count_*sizeof(T);}
    void upload(const T*source,size_t count,size_t offset=0){if(offset+count>count_)throw std::runtime_error("GPU buffer overflow");CUDA_TRY(cudaMemcpy(ptr_+offset,source,count*sizeof(T),cudaMemcpyHostToDevice));}
    void upload(const std::vector<T>&v){upload(v.data(),v.size());}
private:T*ptr_=nullptr;size_t count_=0;
};

class CudaBackendImpl final : public CudaBackend {
public:
    CudaBackendImpl()
    {
        CUDA_TRY(cudaSetDevice(0));
        CUBLAS_TRY(cublasCreate(&cublas_));
        CUDA_TRY(cudaEventCreate(&frame_begin_));
        CUDA_TRY(cudaEventCreate(&frame_end_));
    }

    ~CudaBackendImpl() override
    {
        if(frame_begin_)cudaEventDestroy(frame_begin_);
        if(frame_end_)cudaEventDestroy(frame_end_);
        if(cublas_)cublasDestroy(cublas_);
    }

    KhsResult initialize(const KhsSolverDesc &desc,const KhsHairMaterial &material,
                         const std::vector<CudaNodeInit> &nodes,
                         const std::vector<CudaElementInit> &elements,
                         const std::vector<std::vector<uint32_t>> &strand_nodes,
                         const std::vector<uint32_t> &original_to_internal,
                         const std::vector<KhsVec3> &collider_vertices,
                         const std::vector<CudaTriangleInit> &collider_triangles) override
    {
        try {
            desc_={make_vec(desc.gravity.x,desc.gravity.y,desc.gravity.z),desc.substeps,desc.maximum_substeps,desc.newton_iterations,
                desc.line_search_iterations,desc.absolute_tolerance,desc.relative_tolerance,desc.increment_tolerance,
                desc.minimum_line_search_step,desc.minimum_gap};
            material_={material.density,material.radius,material.young_modulus,material.poisson_ratio,
                material.shear_correction,material.mass_damping,material.contact_stiffness,material.barrier_distance,
                material.friction,material.friction_smoothing,material.collider_offset};
            node_count_=uint32_t(nodes.size());element_count_=uint32_t(elements.size());
            original_count_=uint32_t(original_to_internal.size());collider_count_=uint32_t(collider_vertices.size());
            triangle_count_=uint32_t(collider_triangles.size());strand_count_=uint32_t(strand_nodes.size());dof_count_=size_t(node_count_)*6;

            std::vector<DNode> hn(node_count_);std::vector<double>hq(dof_count_,0.0),hv(dof_count_,0.0);
            std::vector<DVec3>hrp(node_count_),hrr(node_count_);
            for(uint32_t i=0;i<node_count_;++i){const auto&s=nodes[i];DNode&d=hn[i];
                d.reference[0]=make_vec(s.reference[0].x,s.reference[0].y,s.reference[0].z);
                d.reference[1]=make_vec(s.reference[1].x,s.reference[1].y,s.reference[1].z);
                d.reference[2]=make_vec(s.reference[2].x,s.reference[2].y,s.reference[2].z);
                d.mass=s.mass;d.rotational_mass=s.rotational_mass;d.strand=s.strand;d.binding_left=s.binding_left;
                d.binding_right=s.binding_right;d.binding_t=s.binding_t;d.fixed=s.fixed?1u:0u;
                hq[6*i]=s.position.x;hq[6*i+1]=s.position.y;hq[6*i+2]=s.position.z;
                hq[6*i+3]=s.rotation.x;hq[6*i+4]=s.rotation.y;hq[6*i+5]=s.rotation.z;
                hrp[i]=make_vec(s.position.x,s.position.y,s.position.z);hrr[i]=make_vec(s.rotation.x,s.rotation.y,s.rotation.z);
            }
            std::vector<DElement>he(element_count_);for(uint32_t i=0;i<element_count_;++i){const auto&s=elements[i];
                he[i]={s.i,s.j,s.strand,s.collider_contact,s.rest_length,make_vec(s.rest_shear.x,s.rest_shear.y,s.rest_shear.z),
                    make_vec(s.rest_curvature.x,s.rest_curvature.y,s.rest_curvature.z)};}
            maximum_rest_length_=0.0;for(const auto&e:elements)maximum_rest_length_=fmax(maximum_rest_length_,e.rest_length);
            translation_trust_radius_=fmax(0.25*maximum_rest_length_,
                8.0*(material_.radius+material_.collider_offset+material_.barrier_distance));
            std::vector<DStrand>hs(strand_count_);for(uint32_t s=0;s<strand_count_;++s){const auto&indices=strand_nodes[s];uint32_t fixed=0;
                while(fixed<indices.size()&&nodes[indices[fixed]].fixed)++fixed;hs[s]={indices.front(),uint32_t(indices.size()),fixed};}
            std::vector<DVec3>hcv(collider_count_);for(uint32_t i=0;i<collider_count_;++i)hcv[i]=make_vec(collider_vertices[i].x,collider_vertices[i].y,collider_vertices[i].z);
            std::vector<DTriangle>ht(triangle_count_);for(uint32_t i=0;i<triangle_count_;++i)ht[i]={collider_triangles[i].i0,collider_triangles[i].i1,collider_triangles[i].i2};

            host_nodes_=hn;d_nodes_.allocate(node_count_);d_nodes_.upload(host_nodes_);d_elements_.allocate(element_count_);d_elements_.upload(he);d_strands_.allocate(strand_count_);d_strands_.upload(hs);
            d_mapping_.allocate(original_count_);d_mapping_.upload(original_to_internal);
            d_q_.allocate(dof_count_);d_q_.upload(hq.data(),hq.size());d_old_q_.allocate(dof_count_);d_old_q_.upload(hq.data(),hq.size());
            d_snapshot_q_.allocate(dof_count_);d_substep_q_.allocate(dof_count_);d_velocity_.allocate(dof_count_);d_velocity_.upload(hv.data(),hv.size());
            d_snapshot_velocity_.allocate(dof_count_);d_substep_velocity_.allocate(dof_count_);
            d_gradient_.allocate(dof_count_);d_direction_.allocate(dof_count_);d_preconditioner_.allocate(dof_count_);d_base_q_.allocate(dof_count_);
            d_friction_delta_.allocate(dof_count_);
            d_diagonal_blocks_.allocate(size_t(node_count_)*36);d_upper_blocks_.allocate(size_t(node_count_)*36);
            d_cprime_.allocate(size_t(node_count_)*36);d_dprime_.allocate(dof_count_);d_solve_failed_.allocate(1);
            d_root_current_.allocate(node_count_);d_root_current_.upload(hrp);d_rot_current_.allocate(node_count_);d_rot_current_.upload(hrr);
            d_root_pending_.allocate(node_count_);d_root_pending_.upload(hrp);d_rot_pending_.allocate(node_count_);d_rot_pending_.upload(hrr);
            d_output_.allocate(std::max(node_count_,original_count_));d_stats_.allocate(1);d_ccd_limit_.allocate(1);d_sweep_failure_.allocate(1);

            if(collider_count_){d_collider_current_.allocate(collider_count_);d_collider_current_.upload(hcv);d_collider_old_.allocate(collider_count_);d_collider_old_.upload(hcv);
                d_collider_frame_start_.allocate(collider_count_);d_collider_frame_start_.upload(hcv);d_collider_pending_.allocate(collider_count_);d_collider_pending_.upload(hcv);
                d_triangles_.allocate(triangle_count_);d_triangles_.upload(ht);HostBvh bvh;bvh.build(hcv,ht);bvh_levels_=bvh.levels;
                d_bvh_.allocate(bvh.nodes.size());d_bvh_.upload(bvh.nodes);d_order_.allocate(bvh.order.size());d_order_.upload(bvh.order);
                d_leaves_.allocate(bvh.leaves.size());d_leaves_.upload(bvh.leaves);leaf_count_=uint32_t(bvh.leaves.size());
                std::vector<uint32_t>flat;for(const auto&level:bvh.levels){level_offsets_.push_back(uint32_t(flat.size()));flat.insert(flat.end(),level.begin(),level.end());}level_offsets_.push_back(uint32_t(flat.size()));
                d_level_indices_.allocate(flat.size());d_level_indices_.upload(flat);
            }
            clear_failure_diagnostics();update_resident_bytes();
            initialized_=true;
            return KHS_OK;
        } catch(const std::exception&e){error_=std::string("CUDA initialization: ")+e.what();return KHS_ERROR_INTERNAL;}
    }

    KhsResult update_runtime_parameters(const KhsSolverDesc&desc,const KhsHairMaterial&material,
                                        const std::vector<double>&masses,
                                        const std::vector<double>&rotational_masses) override
    {
        try {
            if(!initialized_||masses.size()!=node_count_||rotational_masses.size()!=node_count_)
                return fail(KHS_ERROR_INVALID_STATE,"CUDA runtime parameter update has invalid state.");
            std::vector<DNode>updated=host_nodes_;
            for(uint32_t i=0;i<node_count_;++i){updated[i].mass=masses[i];updated[i].rotational_mass=rotational_masses[i];}
            d_nodes_.upload(updated);
            desc_={make_vec(desc.gravity.x,desc.gravity.y,desc.gravity.z),desc.substeps,desc.maximum_substeps,desc.newton_iterations,
                desc.line_search_iterations,desc.absolute_tolerance,desc.relative_tolerance,desc.increment_tolerance,
                desc.minimum_line_search_step,desc.minimum_gap};
            material_={material.density,material.radius,material.young_modulus,material.poisson_ratio,
                material.shear_correction,material.mass_damping,material.contact_stiffness,material.barrier_distance,
                material.friction,material.friction_smoothing,material.collider_offset};
            host_nodes_=std::move(updated);
            translation_trust_radius_=fmax(0.25*maximum_rest_length_,
                8.0*(material_.radius+material_.collider_offset+material_.barrier_distance));
            error_.clear();return KHS_OK;
        }catch(const std::exception&e){return fail(KHS_ERROR_INTERNAL,std::string("CUDA runtime parameter update: ")+e.what());}
    }

    KhsResult update_roots(const std::vector<KhsVec3>&positions,const std::vector<KhsVec3>&rotations) override
    {
        try {if(positions.size()!=node_count_||rotations.size()!=node_count_)return fail(KHS_ERROR_INVALID_ARGUMENT,"CUDA root target count mismatch.");
            std::vector<DVec3>p(node_count_),r(node_count_);for(uint32_t i=0;i<node_count_;++i){p[i]=make_vec(positions[i].x,positions[i].y,positions[i].z);r[i]=make_vec(rotations[i].x,rotations[i].y,rotations[i].z);}
            d_root_pending_.upload(p);d_rot_pending_.upload(r);return KHS_OK;
        }catch(const std::exception&e){return fail(KHS_ERROR_INTERNAL,e.what());}
    }

    KhsResult update_collider(const KhsVec3*vertices,uint32_t count) override
    {
        try {if(count!=collider_count_)return fail(KHS_ERROR_INVALID_COLLIDER,"CUDA collider vertex count mismatch.");
            std::vector<DVec3>v(count);for(uint32_t i=0;i<count;++i)v[i]=make_vec(vertices[i].x,vertices[i].y,vertices[i].z);d_collider_pending_.upload(v);return KHS_OK;
        }catch(const std::exception&e){return fail(KHS_ERROR_INTERNAL,e.what());}
    }

    KhsResult allocate_animation(uint32_t frame_count) override
    {
        try {if(!initialized_||frame_count<1)return fail(KHS_ERROR_INVALID_ARGUMENT,"Invalid CUDA animation frame count.");
            animation_frames_=frame_count;d_root_animation_.allocate(size_t(frame_count)*node_count_);d_rot_animation_.allocate(size_t(frame_count)*node_count_);
            if(collider_count_)d_collider_animation_.allocate(size_t(frame_count)*collider_count_);root_frame_set_.assign(frame_count,false);collider_frame_set_.assign(frame_count,collider_count_==0);
            gpu_stats_.animation_frame_count=frame_count;update_resident_bytes();return KHS_OK;
        }catch(const std::exception&e){return fail(KHS_ERROR_OUT_OF_MEMORY,e.what());}
    }

    KhsResult set_root_animation_frame(uint32_t frame,const std::vector<KhsVec3>&positions,const std::vector<KhsVec3>&rotations) override
    {
        try {if(frame>=animation_frames_||positions.size()!=node_count_||rotations.size()!=node_count_)return fail(KHS_ERROR_INVALID_ARGUMENT,"Invalid CUDA root animation frame.");
            std::vector<DVec3>p(node_count_),r(node_count_);for(uint32_t i=0;i<node_count_;++i){p[i]=make_vec(positions[i].x,positions[i].y,positions[i].z);r[i]=make_vec(rotations[i].x,rotations[i].y,rotations[i].z);}
            d_root_animation_.upload(p.data(),p.size(),size_t(frame)*node_count_);d_rot_animation_.upload(r.data(),r.size(),size_t(frame)*node_count_);root_frame_set_[frame]=true;return KHS_OK;
        }catch(const std::exception&e){return fail(KHS_ERROR_INTERNAL,e.what());}
    }

    KhsResult set_collider_animation_frame(uint32_t frame,const KhsVec3*vertices,uint32_t count) override
    {
        try {if(frame>=animation_frames_||count!=collider_count_)return fail(KHS_ERROR_INVALID_COLLIDER,"Invalid CUDA collider animation frame.");
            std::vector<DVec3>v(count);for(uint32_t i=0;i<count;++i)v[i]=make_vec(vertices[i].x,vertices[i].y,vertices[i].z);
            d_collider_animation_.upload(v.data(),v.size(),size_t(frame)*collider_count_);collider_frame_set_[frame]=true;return KHS_OK;
        }catch(const std::exception&e){return fail(KHS_ERROR_INTERNAL,e.what());}
    }

    KhsResult finalize_animation() override
    {
        if(animation_frames_<1||std::find(root_frame_set_.begin(),root_frame_set_.end(),false)!=root_frame_set_.end()||
           std::find(collider_frame_set_.begin(),collider_frame_set_.end(),false)!=collider_frame_set_.end())
            return fail(KHS_ERROR_INVALID_STATE,"CUDA animation has missing frames.");
        animation_ready_=true;current_animation_frame_=0;
        CUDA_TRY(cudaMemcpy(d_q_.get(),d_q_.get(),0,cudaMemcpyDeviceToDevice));
        return KHS_OK;
    }

    KhsResult step(double frame_dt) override
    {
        return run_frame(frame_dt,d_root_current_.get(),d_rot_current_.get(),d_root_pending_.get(),d_rot_pending_.get(),
                         d_collider_frame_start_.get(),d_collider_pending_.get(),0,0);
    }

    KhsResult step_animation_frame(uint32_t frame,double frame_dt) override
    {
        if(!animation_ready_||frame>=animation_frames_)return fail(KHS_ERROR_INVALID_ARGUMENT,"Invalid CUDA animation frame index.");
        if(frame==0){current_animation_frame_=0;return KHS_OK;}if(frame!=current_animation_frame_+1)return fail(KHS_ERROR_INVALID_STATE,"CUDA animation frames must be stepped in order.");
        const uint32_t a=frame-1,b=frame;KhsResult r=run_frame(frame_dt,
            d_root_animation_.get()+size_t(a)*node_count_,d_rot_animation_.get()+size_t(a)*node_count_,
            d_root_animation_.get()+size_t(b)*node_count_,d_rot_animation_.get()+size_t(b)*node_count_,
            collider_count_?d_collider_animation_.get()+size_t(a)*collider_count_:nullptr,
            collider_count_?d_collider_animation_.get()+size_t(b)*collider_count_:nullptr,frame,animation_frames_);
        if(r==KHS_OK)current_animation_frame_=frame;return r;
    }

    uint64_t animation_checkpoint_size()const override
    {
        if(!initialized_||!animation_ready_)return 0;
        return uint64_t(sizeof(AnimationCheckpointHeader))+uint64_t(2*dof_count_*sizeof(double));
    }

    KhsResult save_animation_checkpoint(void*data,uint64_t capacity) override
    {
        try {
            const uint64_t required=animation_checkpoint_size();
            if(!data||!required||capacity<required)return fail(KHS_ERROR_INVALID_ARGUMENT,"Invalid CUDA animation checkpoint output buffer.");
            AnimationCheckpointHeader header{kCheckpointMagic,KHS_ABI_VERSION,current_animation_frame_,uint64_t(dof_count_)};
            auto*out=static_cast<unsigned char*>(data);std::memcpy(out,&header,sizeof(header));
            CUDA_TRY(cudaMemcpy(out+sizeof(header),d_q_.get(),d_q_.bytes(),cudaMemcpyDeviceToHost));
            CUDA_TRY(cudaMemcpy(out+sizeof(header)+d_q_.bytes(),d_velocity_.get(),d_velocity_.bytes(),cudaMemcpyDeviceToHost));
            error_.clear();return KHS_OK;
        }catch(const std::exception&e){return fail(KHS_ERROR_INTERNAL,std::string("CUDA checkpoint save: ")+e.what());}
    }

    KhsResult restore_animation_checkpoint(const void*data,uint64_t size) override
    {
        try {
            const uint64_t required=animation_checkpoint_size();
            if(!data||!required||size!=required)return fail(KHS_ERROR_INVALID_ARGUMENT,"Invalid CUDA animation checkpoint size.");
            AnimationCheckpointHeader header{};std::memcpy(&header,data,sizeof(header));
            if(header.magic!=kCheckpointMagic||header.abi_version!=KHS_ABI_VERSION||header.dof_count!=dof_count_||header.animation_frame>=animation_frames_)
                return fail(KHS_ERROR_INVALID_ARGUMENT,"CUDA animation checkpoint does not match this solver.");
            const auto*in=static_cast<const unsigned char*>(data);
            CUDA_TRY(cudaMemcpy(d_q_.get(),in+sizeof(header),d_q_.bytes(),cudaMemcpyHostToDevice));
            CUDA_TRY(cudaMemcpy(d_velocity_.get(),in+sizeof(header)+d_q_.bytes(),d_velocity_.bytes(),cudaMemcpyHostToDevice));
            CUDA_TRY(cudaMemcpy(d_old_q_.get(),d_q_.get(),d_q_.bytes(),cudaMemcpyDeviceToDevice));
            CUDA_TRY(cudaMemcpy(d_snapshot_q_.get(),d_q_.get(),d_q_.bytes(),cudaMemcpyDeviceToDevice));
            CUDA_TRY(cudaMemcpy(d_snapshot_velocity_.get(),d_velocity_.get(),d_velocity_.bytes(),cudaMemcpyDeviceToDevice));
            current_animation_frame_=header.animation_frame;cancel_.store(false);progress_frame_.store(header.animation_frame);
            progress_substep_.store(0);progress_substep_count_.store(0);progress_iteration_.store(0);progress_phase_.store(KHS_PHASE_FINISHED);
            step_stats_={};step_stats_.struct_size=sizeof(step_stats_);step_stats_.phase=KHS_PHASE_FINISHED;clear_failure_diagnostics();error_.clear();return KHS_OK;
        }catch(const std::exception&e){return fail(KHS_ERROR_INTERNAL,std::string("CUDA checkpoint restore: ")+e.what());}
    }

    KhsResult copy_original_positions(KhsVec3*positions,uint32_t capacity)const override
    {
        if(!positions||capacity<original_count_)return KHS_ERROR_INVALID_ARGUMENT;try{gather_original_kernel<<<blocks(original_count_),kThreads>>>(d_q_.get(),d_mapping_.get(),original_count_,d_output_.get());CUDA_TRY(cudaGetLastError());
            std::vector<DVec3>h(original_count_);CUDA_TRY(cudaMemcpy(h.data(),d_output_.get(),h.size()*sizeof(DVec3),cudaMemcpyDeviceToHost));for(uint32_t i=0;i<original_count_;++i)positions[i]={h[i].x,h[i].y,h[i].z};return KHS_OK;
        }catch(...){return KHS_ERROR_INTERNAL;}
    }

    KhsResult copy_internal_positions(KhsVec3*positions,uint32_t capacity)const override
    {
        if(!positions||capacity<node_count_)return KHS_ERROR_INVALID_ARGUMENT;try{gather_internal_kernel<<<blocks(node_count_),kThreads>>>(d_q_.get(),node_count_,d_output_.get());CUDA_TRY(cudaGetLastError());
            std::vector<DVec3>h(node_count_);CUDA_TRY(cudaMemcpy(h.data(),d_output_.get(),h.size()*sizeof(DVec3),cudaMemcpyDeviceToHost));for(uint32_t i=0;i<node_count_;++i)positions[i]={h[i].x,h[i].y,h[i].z};return KHS_OK;
        }catch(...){return KHS_ERROR_INTERNAL;}
    }

    KhsStepStats stats()const override{return step_stats_;}
    KhsGpuStats gpu_stats()const override{return gpu_stats_;}
    std::string last_error()const override{return error_;}
    void request_cancel() override {cancel_.store(true,std::memory_order_relaxed);}
    KhsProgress progress()const override {
        KhsProgress p{};p.struct_size=sizeof(p);p.phase=progress_phase_.load();p.frame_index=progress_frame_.load();p.frame_count=progress_frame_count_.load();
        p.substep=progress_substep_.load();p.substep_count=progress_substep_count_.load();p.nonlinear_iteration=progress_iteration_.load();p.nonlinear_iteration_limit=desc_.newton_iterations;p.cancelled=cancel_.load()?1u:0u;
        p.frame_elapsed_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-frame_clock_).count();return p;
    }
    KhsFailureDiagnostics failure_diagnostics()const override{return failure_diagnostics_;}

private:
    static uint32_t blocks(uint32_t count){return (count+kThreads-1)/kThreads;}
    KhsResult fail(KhsResult result,std::string message){error_=std::move(message);step_stats_.phase=KHS_PHASE_FAILED;progress_phase_.store(KHS_PHASE_FAILED);return result;}

    void clear_failure_diagnostics()
    {
        failure_diagnostics_={};failure_diagnostics_.struct_size=sizeof(failure_diagnostics_);
        failure_diagnostics_.strand_index=std::numeric_limits<uint32_t>::max();
        failure_diagnostics_.element_index=std::numeric_limits<uint32_t>::max();
        failure_diagnostics_.collider_triangle_index=std::numeric_limits<uint32_t>::max();
    }

    void set_failure_context(uint32_t kind,uint32_t frame,uint32_t substep,
                             uint32_t attempted,uint32_t maximum,uint32_t attempts)
    {
        failure_diagnostics_.kind=kind;failure_diagnostics_.frame_index=frame;
        failure_diagnostics_.substep=substep;failure_diagnostics_.requested_substeps=desc_.substeps;
        failure_diagnostics_.attempted_substeps=attempted;failure_diagnostics_.maximum_substeps=maximum;
        failure_diagnostics_.adaptive_attempt_count=attempts;
    }

    void update_resident_bytes()
    {
        gpu_stats_.struct_size=sizeof(gpu_stats_);gpu_stats_.resident_bytes=
            d_nodes_.bytes()+d_elements_.bytes()+d_strands_.bytes()+d_mapping_.bytes()+d_q_.bytes()+d_old_q_.bytes()+d_snapshot_q_.bytes()+d_substep_q_.bytes()+
            d_velocity_.bytes()+d_snapshot_velocity_.bytes()+d_substep_velocity_.bytes()+d_gradient_.bytes()+d_direction_.bytes()+d_preconditioner_.bytes()+d_base_q_.bytes()+d_friction_delta_.bytes()+
            d_diagonal_blocks_.bytes()+d_upper_blocks_.bytes()+d_cprime_.bytes()+d_dprime_.bytes()+
            d_root_current_.bytes()+d_rot_current_.bytes()+d_root_pending_.bytes()+d_rot_pending_.bytes()+
            d_root_animation_.bytes()+d_rot_animation_.bytes()+d_collider_current_.bytes()+d_collider_old_.bytes()+d_collider_frame_start_.bytes()+
            d_collider_pending_.bytes()+d_collider_animation_.bytes()+d_triangles_.bytes()+d_bvh_.bytes()+d_order_.bytes()+d_level_indices_.bytes()+d_leaves_.bytes()+d_output_.bytes()+d_sweep_failure_.bytes();
    }

    void refit_bvh(bool swept)
    {
        if(!collider_count_)return;refit_leaves_kernel<<<blocks(leaf_count_),kThreads>>>(d_bvh_.get(),d_leaves_.get(),leaf_count_,d_order_.get(),d_triangles_.get(),
            d_collider_current_.get(),d_collider_old_.get(),swept);CUDA_TRY(cudaGetLastError());
        for(int depth=int(bvh_levels_.size())-2;depth>=0;--depth){const uint32_t offset=level_offsets_[depth],count=level_offsets_[depth+1]-offset;
            refit_level_kernel<<<blocks(count),kThreads>>>(d_bvh_.get(),d_level_indices_.get()+offset,count);CUDA_TRY(cudaGetLastError());}
    }

    DAssemblyStats evaluate(bool gradient,double dt)
    {
        if(gradient){CUDA_TRY(cudaMemset(d_gradient_.get(),0,d_gradient_.bytes()));CUDA_TRY(cudaMemset(d_upper_blocks_.get(),0,d_upper_blocks_.bytes()));}reset_stats_kernel<<<1,1>>>(d_stats_.get());
        node_energy_kernel<<<blocks(node_count_),kThreads>>>(d_q_.get(),d_old_q_.get(),d_velocity_.get(),d_nodes_.get(),node_count_,desc_,material_,dt,
            gradient?d_gradient_.get():nullptr,gradient?d_diagonal_blocks_.get():nullptr,d_stats_.get());
        element_energy_kernel<<<blocks(element_count_),kThreads>>>(d_q_.get(),d_old_q_.get(),d_nodes_.get(),d_elements_.get(),element_count_,material_,desc_,dt,
            d_collider_current_.get(),d_collider_old_.get(),d_triangles_.get(),d_bvh_.get(),d_order_.get(),collider_count_!=0,gradient?d_gradient_.get():nullptr,
            gradient?d_diagonal_blocks_.get():nullptr,gradient?d_upper_blocks_.get():nullptr,d_stats_.get());
        CUDA_TRY(cudaGetLastError());DAssemblyStats out{};CUDA_TRY(cudaMemcpy(&out,d_stats_.get(),sizeof(out),cudaMemcpyDeviceToHost));return out;
    }

    void apply_friction(double dt)
    {
        if(!collider_count_||!(material_.friction>0.0))return;CUDA_TRY(cudaMemset(d_friction_delta_.get(),0,d_friction_delta_.bytes()));
        friction_impulse_kernel<<<blocks(element_count_),kThreads>>>(d_q_.get(),d_friction_delta_.get(),d_velocity_.get(),d_nodes_.get(),d_elements_.get(),element_count_,
            d_collider_current_.get(),d_collider_old_.get(),d_triangles_.get(),d_bvh_.get(),d_order_.get(),material_,dt);CUDA_TRY(cudaGetLastError());
        const double one=1.0;CUBLAS_TRY(cublasDaxpy(cublas_,int(dof_count_),&one,d_friction_delta_.get(),1,d_velocity_.get(),1));
    }

    void limit_direction()
    {
        double maximum_translation=0.0;CUDA_TRY(cudaMemcpy(d_ccd_limit_.get(),&maximum_translation,sizeof(double),cudaMemcpyHostToDevice));
        maximum_translation_kernel<<<blocks(node_count_),kThreads>>>(d_direction_.get(),d_nodes_.get(),node_count_,d_ccd_limit_.get());
        CUDA_TRY(cudaMemcpy(&maximum_translation,d_ccd_limit_.get(),sizeof(double),cudaMemcpyDeviceToHost));
        if(maximum_translation>translation_trust_radius_){const double scale=translation_trust_radius_/maximum_translation;
            CUBLAS_TRY(cublasDscal(cublas_,int(dof_count_),&scale,d_direction_.get(),1));}
    }

    void make_steepest_direction()
    {
        CUBLAS_TRY(cublasDcopy(cublas_,int(dof_count_),d_gradient_.get(),1,d_direction_.get(),1));const double neg=-1.0;
        CUBLAS_TRY(cublasDscal(cublas_,int(dof_count_),&neg,d_direction_.get(),1));
        multiply_kernel<<<blocks(uint32_t(dof_count_)),kThreads>>>(d_direction_.get(),d_preconditioner_.get(),uint32_t(dof_count_));
        enforce_fixed_direction_kernel<<<blocks(node_count_),kThreads>>>(d_direction_.get(),d_nodes_.get(),node_count_);limit_direction();
    }

    bool make_direction()
    {
        CUDA_TRY(cudaMemset(d_direction_.get(),0,d_direction_.bytes()));int failed=0;CUDA_TRY(cudaMemcpy(d_solve_failed_.get(),&failed,sizeof(int),cudaMemcpyHostToDevice));
        block_solve_kernel<<<blocks(strand_count_),kThreads>>>(d_strands_.get(),strand_count_,d_diagonal_blocks_.get(),d_upper_blocks_.get(),d_gradient_.get(),
            d_cprime_.get(),d_dprime_.get(),d_direction_.get(),d_solve_failed_.get());CUDA_TRY(cudaMemcpy(&failed,d_solve_failed_.get(),sizeof(int),cudaMemcpyDeviceToHost));
        double slope=0;bool used_block_solve=false;if(!failed){CUBLAS_TRY(cublasDdot(cublas_,int(dof_count_),d_gradient_.get(),1,d_direction_.get(),1,&slope));used_block_solve=slope<0&&std::isfinite(slope);}
        if(!used_block_solve){make_steepest_direction();return false;}limit_direction();
        return used_block_solve;
    }

    KhsResult optimize_substep(double dt)
    {
        preconditioner_node_kernel<<<blocks(node_count_),kThreads>>>(d_preconditioner_.get(),d_nodes_.get(),node_count_,material_,dt);
        preconditioner_element_kernel<<<blocks(element_count_),kThreads>>>(d_preconditioner_.get(),d_elements_.get(),element_count_,material_);
        preconditioner_finish_kernel<<<blocks(node_count_),kThreads>>>(d_preconditioner_.get(),d_nodes_.get(),node_count_);
        DAssemblyStats current=evaluate(true,dt);if(!current.feasible)return fail(KHS_ERROR_NOT_CONVERGED,"CUDA state is outside the barrier feasible region.");
        double initial_norm=0;CUBLAS_TRY(cublasDnrm2(cublas_,int(dof_count_),d_gradient_.get(),1,&initial_norm));step_stats_.initial_residual_norm=fmax(step_stats_.initial_residual_norm,initial_norm);
        for(uint32_t iteration=0;iteration<=desc_.newton_iterations;++iteration){progress_iteration_.store(iteration);if(cancel_.load())return fail(KHS_ERROR_INVALID_STATE,"CUDA solve cancelled.");
            double residual=0;CUBLAS_TRY(cublasDnrm2(cublas_,int(dof_count_),d_gradient_.get(),1,&residual));const double relative=residual/fmax(initial_norm,1e-30);
            step_stats_.final_residual_norm=residual;step_stats_.relative_residual_norm=relative;step_stats_.minimum_gap=fmin(step_stats_.minimum_gap,current.minimum_gap);
            step_stats_.contact_candidate_count+=current.candidates;step_stats_.active_contact_count+=current.active;
            step_stats_.kinetic_energy=current.kinetic;step_stats_.elastic_energy=current.elastic;step_stats_.contact_energy=current.contact;step_stats_.friction_energy=current.friction;
            const double gpu_absolute_floor=1.0e-5*sqrt(double(dof_count_));
            if(residual<=fmax(desc_.absolute_tolerance,gpu_absolute_floor)||relative<=fmax(desc_.relative_tolerance,1.0e-4)){update_velocity_kernel<<<blocks(node_count_),kThreads>>>(d_q_.get(),d_old_q_.get(),d_velocity_.get(),d_nodes_.get(),node_count_,1.0/dt);return KHS_OK;}
            if(iteration==desc_.newton_iterations)break;progress_phase_.store(KHS_PHASE_LINEAR_SOLVE);bool block_direction=make_direction();++step_stats_.linear_solves;
            CUBLAS_TRY(cublasDcopy(cublas_,int(dof_count_),d_q_.get(),1,d_base_q_.get(),1));
            bool accepted=false;double alpha=1.0;DAssemblyStats trial{};
            for(int direction_attempt=0;direction_attempt<2;++direction_attempt){
                double slope=0;CUBLAS_TRY(cublasDdot(cublas_,int(dof_count_),d_gradient_.get(),1,d_direction_.get(),1,&slope));alpha=1.0;
                if(collider_count_){progress_phase_.store(KHS_PHASE_CCD);CUDA_TRY(cudaMemcpy(d_ccd_limit_.get(),&alpha,sizeof(double),cudaMemcpyHostToDevice));
                    ccd_limit_kernel<<<blocks(element_count_),kThreads>>>(d_q_.get(),d_direction_.get(),d_nodes_.get(),d_elements_.get(),element_count_,d_collider_current_.get(),d_triangles_.get(),d_bvh_.get(),d_order_.get(),material_,desc_,d_ccd_limit_.get());
                    CUDA_TRY(cudaMemcpy(&alpha,d_ccd_limit_.get(),sizeof(double),cudaMemcpyDeviceToHost));if(alpha<1.0)alpha*=0.9;step_stats_.ccd_step_limit=fmin(step_stats_.ccd_step_limit,alpha);}
                accepted=false;progress_phase_.store(KHS_PHASE_LINE_SEARCH);
                if(alpha>=desc_.minimum_line_search_step)for(uint32_t search=0;search<desc_.line_search_iterations;++search){CUBLAS_TRY(cublasDcopy(cublas_,int(dof_count_),d_base_q_.get(),1,d_q_.get(),1));CUBLAS_TRY(cublasDaxpy(cublas_,int(dof_count_),&alpha,d_direction_.get(),1,d_q_.get(),1));
                    trial=evaluate(false,dt);++step_stats_.line_search_evaluations;if(trial.feasible&&std::isfinite(trial.objective)&&trial.objective<=current.objective+1e-4*alpha*slope){accepted=true;break;}
                    alpha*=0.5;if(alpha<desc_.minimum_line_search_step)break;}
                if(accepted)break;
                if(!block_direction)break;CUBLAS_TRY(cublasDcopy(cublas_,int(dof_count_),d_base_q_.get(),1,d_q_.get(),1));make_steepest_direction();block_direction=false;
            }
            if(!accepted){CUBLAS_TRY(cublasDcopy(cublas_,int(dof_count_),d_base_q_.get(),1,d_q_.get(),1));return fail(KHS_ERROR_NOT_CONVERGED,"CUDA Gauss-Newton line search failed.");}
            current=evaluate(true,dt);double direction_norm=0;CUBLAS_TRY(cublasDnrm2(cublas_,int(dof_count_),d_direction_.get(),1,&direction_norm));
            step_stats_.increment_norm=alpha*direction_norm;step_stats_.accepted_step_length=alpha;
            ++step_stats_.newton_iterations;progress_phase_.store(KHS_PHASE_ASSEMBLING);
            const double rms_increment=step_stats_.increment_norm/sqrt(double(dof_count_));
            if(rms_increment<=2.0*desc_.increment_tolerance){update_velocity_kernel<<<blocks(node_count_),kThreads>>>(d_q_.get(),d_old_q_.get(),d_velocity_.get(),d_nodes_.get(),node_count_,1.0/dt);return KHS_OK;}
        }
        update_velocity_kernel<<<blocks(node_count_),kThreads>>>(d_q_.get(),d_old_q_.get(),d_velocity_.get(),d_nodes_.get(),node_count_,1.0/dt);
        return KHS_OK;
    }

    KhsResult run_frame(double frame_dt,const DVec3*root_a,const DVec3*rot_a,const DVec3*root_b,const DVec3*rot_b,
                        const DVec3*collider_a,const DVec3*collider_b,uint32_t frame,uint32_t frame_count)
    {
        if(!(frame_dt>0&&std::isfinite(frame_dt)))return fail(KHS_ERROR_INVALID_ARGUMENT,"Invalid CUDA time step.");cancel_.store(false);clear_failure_diagnostics();frame_clock_=std::chrono::steady_clock::now();progress_frame_.store(frame);progress_frame_count_.store(frame_count);progress_iteration_.store(0);
        CUDA_TRY(cudaEventRecord(frame_begin_));CUDA_TRY(cudaMemcpy(d_snapshot_q_.get(),d_q_.get(),d_q_.bytes(),cudaMemcpyDeviceToDevice));CUDA_TRY(cudaMemcpy(d_snapshot_velocity_.get(),d_velocity_.get(),d_velocity_.bytes(),cudaMemcpyDeviceToDevice));
        const uint32_t maximum_substeps=collider_count_?desc_.maximum_substeps:desc_.substeps;
        uint32_t accepted_substeps=0,attempted_substeps=0,toi_limited_substeps=0;
        double completed_fraction=0.0;
        DMovingSweepFailure last_sweep{};double last_attempt_span=0.0,last_safe_limit=1.0;
        step_stats_={};step_stats_.struct_size=sizeof(step_stats_);step_stats_.minimum_gap=std::numeric_limits<double>::infinity();step_stats_.ccd_step_limit=1.0;
        progress_substep_count_.store(maximum_substeps);progress_iteration_.store(0);progress_phase_.store(KHS_PHASE_ASSEMBLING);
        if(collider_count_){CUDA_TRY(cudaMemcpy(d_collider_current_.get(),collider_a,d_collider_current_.bytes(),cudaMemcpyDeviceToDevice));CUDA_TRY(cudaMemcpy(d_collider_frame_start_.get(),collider_a,d_collider_current_.bytes(),cudaMemcpyDeviceToDevice));}

        auto restore_frame=[&](){restore_snapshot();if(collider_count_){CUDA_TRY(cudaMemcpy(d_collider_current_.get(),collider_a,d_collider_current_.bytes(),cudaMemcpyDeviceToDevice));CUDA_TRY(cudaMemcpy(d_collider_old_.get(),collider_a,d_collider_old_.bytes(),cudaMemcpyDeviceToDevice));refit_bvh(false);}};
        auto record_failure=[&](uint32_t kind,uint32_t substep){
            set_failure_context(kind,frame,substep,attempted_substeps,maximum_substeps,toi_limited_substeps);
            if(last_sweep.claimed){failure_diagnostics_.strand_index=last_sweep.strand_index;failure_diagnostics_.element_index=last_sweep.element_index;failure_diagnostics_.collider_triangle_index=last_sweep.triangle_index;
                failure_diagnostics_.distance=last_sweep.distance;failure_diagnostics_.required_distance=last_sweep.required_distance;failure_diagnostics_.clearance=last_sweep.clearance;
                failure_diagnostics_.collider_substep_displacement=last_sweep.collider_displacement*last_safe_limit;
                failure_diagnostics_.collider_frame_displacement=last_attempt_span>0.0?last_sweep.collider_displacement/last_attempt_span:0.0;
                failure_diagnostics_.hair_start={last_sweep.hair_start.x,last_sweep.hair_start.y,last_sweep.hair_start.z};failure_diagnostics_.hair_end={last_sweep.hair_end.x,last_sweep.hair_end.y,last_sweep.hair_end.z};failure_diagnostics_.collider_point={last_sweep.collider_point.x,last_sweep.collider_point.y,last_sweep.collider_point.z};}
        };

        for(uint32_t nominal=1;nominal<=desc_.substeps;++nominal){
            const double nominal_end=double(nominal)/double(desc_.substeps);double retry_end=nominal_end;
            while(completed_fraction+desc_.minimum_line_search_step<nominal_end){
                if(cancel_.load()){restore_frame();return fail(KHS_ERROR_INVALID_STATE,"CUDA solve cancelled.");}
                if(attempted_substeps>=maximum_substeps){record_failure(toi_limited_substeps?KHS_FAILURE_MOVING_COLLIDER_SWEEP:KHS_FAILURE_NONLINEAR_SOLVE,accepted_substeps+1);restore_frame();return fail(KHS_ERROR_NOT_CONVERGED,"Animated collider exhausted the variable CUDA time-step budget.");}
                ++attempted_substeps;progress_substep_.store(accepted_substeps+1);progress_iteration_.store(0);error_.clear();progress_phase_.store(KHS_PHASE_CCD);
                const double proposed_end=fmin(nominal_end,retry_end);const double attempted_span=proposed_end-completed_fraction;
                if(!(attempted_span>desc_.minimum_line_search_step)){record_failure(KHS_FAILURE_MOVING_COLLIDER_SWEEP,accepted_substeps+1);restore_frame();return fail(KHS_ERROR_NOT_CONVERGED,"Animated collider TOI collapsed below the minimum CUDA time step.");}

                double safe_limit=1.0;DMovingSweepFailure sweep{};
                if(collider_count_){CUDA_TRY(cudaMemcpy(d_collider_old_.get(),d_collider_current_.get(),d_collider_current_.bytes(),cudaMemcpyDeviceToDevice));interpolate_collider_kernel<<<blocks(collider_count_),kThreads>>>(d_collider_current_.get(),collider_a,collider_b,collider_count_,proposed_end);
                    refit_bvh(true);CUDA_TRY(cudaMemcpy(d_ccd_limit_.get(),&safe_limit,sizeof(double),cudaMemcpyHostToDevice));CUDA_TRY(cudaMemcpy(d_sweep_failure_.get(),&sweep,sizeof(sweep),cudaMemcpyHostToDevice));
                    moving_sweep_kernel<<<blocks(element_count_),kThreads>>>(d_q_.get(),d_nodes_.get(),d_elements_.get(),element_count_,d_collider_old_.get(),d_collider_current_.get(),d_triangles_.get(),d_bvh_.get(),d_order_.get(),material_,desc_,d_ccd_limit_.get(),d_sweep_failure_.get());
                    CUDA_TRY(cudaMemcpy(&safe_limit,d_ccd_limit_.get(),sizeof(double),cudaMemcpyDeviceToHost));if(safe_limit<1.0){CUDA_TRY(cudaMemcpy(&sweep,d_sweep_failure_.get(),sizeof(sweep),cudaMemcpyDeviceToHost));safe_limit=fmax(0.0,fmin(1.0,safe_limit));++toi_limited_substeps;step_stats_.ccd_step_limit=fmin(step_stats_.ccd_step_limit,safe_limit);last_sweep=sweep;last_attempt_span=attempted_span;last_safe_limit=safe_limit;}
                }
                const double actual_end=completed_fraction+attempted_span*safe_limit;const double actual_span=actual_end-completed_fraction;
                if(!(actual_span>desc_.minimum_line_search_step)){record_failure(KHS_FAILURE_MOVING_COLLIDER_SWEEP,accepted_substeps+1);restore_frame();return fail(KHS_ERROR_NOT_CONVERGED,"Animated collider TOI collapsed below the minimum CUDA time step.");}
                if(collider_count_&&safe_limit<1.0){interpolate_collider_kernel<<<blocks(collider_count_),kThreads>>>(d_collider_current_.get(),collider_a,collider_b,collider_count_,actual_end);refit_bvh(true);}

                CUDA_TRY(cudaMemcpy(d_substep_q_.get(),d_q_.get(),d_q_.bytes(),cudaMemcpyDeviceToDevice));CUDA_TRY(cudaMemcpy(d_substep_velocity_.get(),d_velocity_.get(),d_velocity_.bytes(),cudaMemcpyDeviceToDevice));CUDA_TRY(cudaMemcpy(d_old_q_.get(),d_q_.get(),d_q_.bytes(),cudaMemcpyDeviceToDevice));
                const double dt=frame_dt*actual_span;begin_substep_kernel<<<blocks(node_count_),kThreads>>>(d_q_.get(),d_old_q_.get(),d_velocity_.get(),d_nodes_.get(),node_count_,root_a,rot_a,root_b,rot_b,actual_end,dt);
                if(collider_count_)refit_bvh(false);progress_phase_.store(KHS_PHASE_ASSEMBLING);const KhsResult attempt_result=optimize_substep(dt);
                if(attempt_result!=KHS_OK){CUDA_TRY(cudaMemcpy(d_q_.get(),d_substep_q_.get(),d_q_.bytes(),cudaMemcpyDeviceToDevice));CUDA_TRY(cudaMemcpy(d_velocity_.get(),d_substep_velocity_.get(),d_velocity_.bytes(),cudaMemcpyDeviceToDevice));if(collider_count_){CUDA_TRY(cudaMemcpy(d_collider_current_.get(),d_collider_old_.get(),d_collider_current_.bytes(),cudaMemcpyDeviceToDevice));refit_bvh(false);}
                    if(cancel_.load()){restore_frame();return attempt_result;}retry_end=completed_fraction+0.5*actual_span;if(!(retry_end-completed_fraction>desc_.minimum_line_search_step)||attempted_substeps>=maximum_substeps){const uint32_t kind=error_.find("outside the barrier")!=std::string::npos?KHS_FAILURE_BARRIER_INFEASIBLE:KHS_FAILURE_NONLINEAR_SOLVE;record_failure(kind,accepted_substeps+1);restore_frame();return attempt_result;}continue;}
                apply_friction(dt);++accepted_substeps;++step_stats_.converged_substeps;step_stats_.substeps=accepted_substeps;completed_fraction=actual_end;retry_end=nominal_end;
            }
            completed_fraction=nominal_end;
        }
        if(!animation_ready_){CUDA_TRY(cudaMemcpy(d_root_current_.get(),d_root_pending_.get(),d_root_current_.bytes(),cudaMemcpyDeviceToDevice));CUDA_TRY(cudaMemcpy(d_rot_current_.get(),d_rot_pending_.get(),d_rot_current_.bytes(),cudaMemcpyDeviceToDevice));
            if(collider_count_)CUDA_TRY(cudaMemcpy(d_collider_frame_start_.get(),d_collider_pending_.get(),d_collider_frame_start_.bytes(),cudaMemcpyDeviceToDevice));}
        clear_failure_diagnostics();step_stats_.phase=KHS_PHASE_FINISHED;progress_phase_.store(KHS_PHASE_FINISHED);CUDA_TRY(cudaEventRecord(frame_end_));CUDA_TRY(cudaEventSynchronize(frame_end_));float ms=0;CUDA_TRY(cudaEventElapsedTime(&ms,frame_begin_,frame_end_));gpu_stats_.last_frame_milliseconds=ms;error_.clear();return KHS_OK;
    }

    void restore_snapshot(){CUDA_TRY(cudaMemcpy(d_q_.get(),d_snapshot_q_.get(),d_q_.bytes(),cudaMemcpyDeviceToDevice));CUDA_TRY(cudaMemcpy(d_velocity_.get(),d_snapshot_velocity_.get(),d_velocity_.bytes(),cudaMemcpyDeviceToDevice));}

    cublasHandle_t cublas_=nullptr;cudaEvent_t frame_begin_=nullptr,frame_end_=nullptr;
    DDesc desc_{};DMaterial material_{};uint32_t node_count_=0,element_count_=0,original_count_=0,collider_count_=0,triangle_count_=0,strand_count_=0;size_t dof_count_=0;
    bool initialized_=false,animation_ready_=false;uint32_t animation_frames_=0,current_animation_frame_=0,leaf_count_=0;double translation_trust_radius_=0.0,maximum_rest_length_=0.0;
    std::vector<DNode>host_nodes_;
    std::vector<bool>root_frame_set_,collider_frame_set_;std::vector<std::vector<uint32_t>>bvh_levels_;std::vector<uint32_t>level_offsets_;
    DeviceBuffer<DNode>d_nodes_;DeviceBuffer<DElement>d_elements_;DeviceBuffer<DStrand>d_strands_;DeviceBuffer<uint32_t>d_mapping_;
    DeviceBuffer<double>d_q_,d_old_q_,d_snapshot_q_,d_substep_q_,d_velocity_,d_snapshot_velocity_,d_substep_velocity_,d_gradient_,d_direction_,d_preconditioner_,d_base_q_,d_friction_delta_;
    DeviceBuffer<double>d_diagonal_blocks_,d_upper_blocks_,d_cprime_,d_dprime_;DeviceBuffer<int>d_solve_failed_;
    DeviceBuffer<DVec3>d_root_current_,d_rot_current_,d_root_pending_,d_rot_pending_,d_root_animation_,d_rot_animation_;
    DeviceBuffer<DVec3>d_collider_current_,d_collider_old_,d_collider_frame_start_,d_collider_pending_,d_collider_animation_,d_output_;
    DeviceBuffer<DTriangle>d_triangles_;DeviceBuffer<DBvhNode>d_bvh_;DeviceBuffer<uint32_t>d_order_,d_level_indices_,d_leaves_;
    DeviceBuffer<DAssemblyStats>d_stats_;DeviceBuffer<double>d_ccd_limit_;DeviceBuffer<DMovingSweepFailure>d_sweep_failure_;
    KhsStepStats step_stats_{};KhsGpuStats gpu_stats_{};KhsFailureDiagnostics failure_diagnostics_{};std::string error_;
    std::atomic<bool>cancel_{false};std::atomic<uint32_t>progress_phase_{KHS_PHASE_IDLE},progress_frame_{0},progress_frame_count_{0},progress_substep_{0},progress_substep_count_{0},progress_iteration_{0};
    std::chrono::steady_clock::time_point frame_clock_=std::chrono::steady_clock::now();
};

} // namespace

std::unique_ptr<CudaBackend> make_cuda_backend()
{
    try{return std::make_unique<CudaBackendImpl>();}catch(...){return nullptr;}
}

bool query_cuda_device(KhsGpuInfo&info,std::string&error)
{
    std::memset(&info,0,sizeof(info));info.struct_size=sizeof(info);int count=0;const cudaError_t result=cudaGetDeviceCount(&count);
    if(result!=cudaSuccess||count<1){error=result==cudaSuccess?"No CUDA GPU found.":cudaGetErrorString(result);return false;}
    cudaDeviceProp prop{};if(cudaGetDeviceProperties(&prop,0)!=cudaSuccess){error="Failed to query CUDA device.";return false;}
    info.available=1;info.device_ordinal=0;info.compute_capability_major=prop.major;info.compute_capability_minor=prop.minor;info.total_vram_bytes=prop.totalGlobalMem;
    std::strncpy(info.device_name,prop.name,sizeof(info.device_name)-1);return true;
}

} // namespace kami

namespace kami {
namespace {

__global__ void reset_stats_kernel(DAssemblyStats *s)
{
    if(threadIdx.x||blockIdx.x)return;
    s->objective=s->elastic=s->contact=s->friction=s->kinetic=0.0;
    s->minimum_gap=__longlong_as_double(0x7ff0000000000000ULL);s->candidates=s->active=0;s->feasible=1;
}

__global__ void interpolate_collider_kernel(DVec3 *current,const DVec3 *a,const DVec3 *b,
                                             uint32_t count,double t)
{
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)current[i]=a[i]*(1.0-t)+b[i]*t;
}

__global__ void refit_leaves_kernel(DBvhNode *nodes,const uint32_t *leaf_indices,uint32_t leaf_count,
                                    const uint32_t *order,const DTriangle *triangles,
                                    const DVec3 *vertices,const DVec3 *old_vertices,bool swept)
{
    const uint32_t k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=leaf_count)return;DBvhNode&n=nodes[leaf_indices[k]];
    DVec3 lo=make_vec(1e300,1e300,1e300),hi=make_vec(-1e300,-1e300,-1e300);
    for(uint32_t j=n.begin;j<n.begin+n.count;++j){const DTriangle&t=triangles[order[j]];const uint32_t ids[3]{t.i0,t.i1,t.i2};
        for(int c=0;c<3;++c){lo=min_vec(lo,vertices[ids[c]]);hi=max_vec(hi,vertices[ids[c]]);if(swept){lo=min_vec(lo,old_vertices[ids[c]]);hi=max_vec(hi,old_vertices[ids[c]]);}}}
    n.lower=lo;n.upper=hi;
}

__global__ void refit_level_kernel(DBvhNode *nodes,const uint32_t *indices,uint32_t count)
{
    const uint32_t k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;DBvhNode&n=nodes[indices[k]];
    if(n.left<0)return;
    n.lower=min_vec(nodes[n.left].lower,nodes[n.right].lower);n.upper=max_vec(nodes[n.left].upper,nodes[n.right].upper);
}

__global__ void begin_substep_kernel(double*q,const double*old_q,double*velocity,const DNode*nodes,
                                     uint32_t node_count,const DVec3*root_a,const DVec3*rot_a,
                                     const DVec3*root_b,const DVec3*rot_b,double t,double dt)
{
    const uint32_t n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=node_count||!nodes[n].fixed)return;
    const DVec3 p=root_a[n]*(1.0-t)+root_b[n]*t,r=rot_a[n]*(1.0-t)+rot_b[n]*t;
    const double values[6]{p.x,p.y,p.z,r.x,r.y,r.z};for(int d=0;d<6;++d){q[6*n+d]=values[d];velocity[6*n+d]=(values[d]-old_q[6*n+d])/dt;}
}

__global__ void update_velocity_kernel(const double*q,const double*old_q,double*velocity,
                                       const DNode*nodes,uint32_t node_count,double inv_dt)
{
    const uint32_t n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=node_count||nodes[n].fixed)return;
    for(int d=0;d<6;++d)velocity[6*n+d]=(q[6*n+d]-old_q[6*n+d])*inv_dt;
}

__global__ void enforce_fixed_direction_kernel(double*v,const DNode*nodes,uint32_t node_count)
{
    const uint32_t n=blockIdx.x*blockDim.x+threadIdx.x;if(n<node_count&&nodes[n].fixed)for(int d=0;d<6;++d)v[6*n+d]=0.0;
}

__global__ void preconditioner_node_kernel(double*diagonal,const DNode*nodes,uint32_t node_count,
                                           DMaterial material,double dt)
{
    const uint32_t n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=node_count)return;const double inv=1.0/dt,inv2=inv*inv;
    const double position=nodes[n].mass*(inv2+material.mass_damping*inv);
    const double rotation=nodes[n].rotational_mass*(inv2+material.mass_damping*inv);
    for(int d=0;d<3;++d){diagonal[6*n+d]=position;diagonal[6*n+3+d]=rotation;}
}

__global__ void preconditioner_element_kernel(double*diagonal,const DElement*elements,uint32_t count,DMaterial material)
{
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;const DElement&e=elements[i];
    const double area=kPi*material.radius*material.radius;
    const double inertia=kPi*material.radius*material.radius*material.radius*material.radius/4.0;
    const double shear=material.young_modulus/(2.0*(1.0+material.poisson_ratio));
    const double translation=fmax(material.young_modulus*area,material.shear_correction*shear*area)/e.rest_length;
    const double rotation=material.shear_correction*shear*area*e.rest_length+
                          fmax(material.young_modulus*inertia,2.0*shear*inertia)/e.rest_length;
    for(int d=0;d<3;++d){atomicAdd(&diagonal[6*e.i+d],translation);atomicAdd(&diagonal[6*e.j+d],translation);
        atomicAdd(&diagonal[6*e.i+3+d],rotation);atomicAdd(&diagonal[6*e.j+3+d],rotation);}
}

__global__ void preconditioner_finish_kernel(double*diagonal,const DNode*nodes,uint32_t node_count)
{
    const uint32_t n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=node_count)return;
    for(int d=0;d<6;++d)diagonal[6*n+d]=nodes[n].fixed?0.0:1.0/fmax(diagonal[6*n+d],1.0e-30);
}

__global__ void multiply_kernel(double*values,const double*scale,uint32_t count)
{
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)values[i]*=scale[i];
}

__global__ void maximum_translation_kernel(const double*values,const DNode*nodes,uint32_t count,double*maximum)
{
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count||nodes[i].fixed)return;
    const DVec3 displacement=make_vec(values[6*i],values[6*i+1],values[6*i+2]);
    atomic_max_double(maximum,norm(displacement));
}

__device__ bool inverse6(const double*source,double*inverse)
{
    double a[36];for(int i=0;i<36;++i){a[i]=source[i];inverse[i]=0.0;}for(int i=0;i<6;++i)inverse[6*i+i]=1.0;
    double scale=0.0;for(int i=0;i<6;++i)scale=fmax(scale,fabs(a[6*i+i]));for(int i=0;i<6;++i)a[6*i+i]+=fmax(1.0e-12*scale,1.0e-24);
    for(int col=0;col<6;++col){int pivot=col;double best=fabs(a[6*col+col]);for(int row=col+1;row<6;++row)if(fabs(a[6*row+col])>best){best=fabs(a[6*row+col]);pivot=row;}
        if(best<1.0e-30||!isfinite(best))return false;if(pivot!=col)for(int k=0;k<6;++k){double t=a[6*col+k];a[6*col+k]=a[6*pivot+k];a[6*pivot+k]=t;t=inverse[6*col+k];inverse[6*col+k]=inverse[6*pivot+k];inverse[6*pivot+k]=t;}
        const double invp=1.0/a[6*col+col];for(int k=0;k<6;++k){a[6*col+k]*=invp;inverse[6*col+k]*=invp;}
        for(int row=0;row<6;++row)if(row!=col){const double f=a[6*row+col];for(int k=0;k<6;++k){a[6*row+k]-=f*a[6*col+k];inverse[6*row+k]-=f*inverse[6*col+k];}}}
    return true;
}

__global__ void block_solve_kernel(const DStrand*strands,uint32_t strand_count,const double*diagonal,
                                   const double*upper,const double*gradient,double*cprime,double*dprime,
                                   double*direction,int*failed)
{
    const uint32_t s=blockIdx.x*blockDim.x+threadIdx.x;if(s>=strand_count)return;const DStrand strand=strands[s];
    const uint32_t begin=strand.first+strand.fixed_count,end=strand.first+strand.count;if(begin>=end)return;
    for(uint32_t n=begin;n<end;++n){double b[36],inv[36],rhs[6];for(int i=0;i<36;++i)b[i]=diagonal[36*n+i];for(int i=0;i<6;++i)rhs[i]=-gradient[6*n+i];
        if(n>begin){const double*lower=upper+36*(n-1);const double*previous_c=cprime+36*(n-1);const double*previous_d=dprime+6*(n-1);
            for(int r=0;r<6;++r){for(int c=0;c<6;++c){double sum=0;for(int k=0;k<6;++k)sum+=lower[6*k+r]*previous_c[6*k+c];b[6*r+c]-=sum;}
                double sum=0;for(int k=0;k<6;++k)sum+=lower[6*k+r]*previous_d[k];rhs[r]-=sum;}}
        if(!inverse6(b,inv)){atomicExch(failed,1);return;}double*current_c=cprime+36*n;double*current_d=dprime+6*n;
        if(n+1<end){const double*up=upper+36*n;for(int r=0;r<6;++r)for(int c=0;c<6;++c){double sum=0;for(int k=0;k<6;++k)sum+=inv[6*r+k]*up[6*k+c];current_c[6*r+c]=sum;}}
        else for(int i=0;i<36;++i)current_c[i]=0.0;
        for(int r=0;r<6;++r){double sum=0;for(int k=0;k<6;++k)sum+=inv[6*r+k]*rhs[k];current_d[r]=sum;}}
    for(uint32_t n=end;n-->begin;){for(int r=0;r<6;++r){double value=dprime[6*n+r];if(n+1<end)for(int k=0;k<6;++k)value-=cprime[36*n+6*r+k]*direction[6*(n+1)+k];direction[6*n+r]=value;}}
}

__global__ void node_energy_kernel(const double*q,const double*old_q,const double*velocity,
                                   const DNode*nodes,uint32_t node_count,DDesc desc,DMaterial material,
                                   double dt,double*gradient,double*diagonal,DAssemblyStats*stats)
{
    const uint32_t n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=node_count)return;
    if(diagonal)for(int k=0;k<36;++k)diagonal[36*n+k]=0.0;if(nodes[n].fixed)return;
    const double inv=1.0/dt,inv2=inv*inv,kx=nodes[n].mass*inv2,kr=nodes[n].rotational_mass*inv2;
    const double cx=nodes[n].mass*material.mass_damping*inv,cr=nodes[n].rotational_mass*material.mass_damping*inv;
    double objective=0,kinetic=0;for(int d=0;d<6;++d){const double gravity=d<3?((&desc.gravity.x)[d])*dt*dt:0.0;
        const double predicted=old_q[6*n+d]+dt*velocity[6*n+d]+gravity;const double delta=q[6*n+d]-predicted;
        const double damp=q[6*n+d]-old_q[6*n+d],k=d<3?kx:kr,c=d<3?cx:cr;
        objective+=0.5*k*delta*delta+0.5*c*damp*damp;if(d<3)kinetic+=0.5*nodes[n].mass*damp*damp*inv2;
        if(gradient)gradient[6*n+d]=k*delta+c*damp;if(diagonal)diagonal[36*n+d*6+d]=k+c;}
    atomicAdd(&stats->objective,objective);atomicAdd(&stats->kinetic,kinetic);
}

__global__ void element_energy_kernel(const double*q,const double*old_q,const DNode*nodes,
                                      const DElement*elements,uint32_t element_count,DMaterial material,DDesc desc,
                                      double dt,
                                      const DVec3*collider,const DVec3*old_collider,
                                      const DTriangle*triangles,const DBvhNode*bvh,const uint32_t*order,
                                      bool has_collider,double*gradient,double*diagonal,double*upper,DAssemblyStats*stats)
{
    const uint32_t ei=blockIdx.x*blockDim.x+threadIdx.x;if(ei>=element_count)return;const DElement&e=elements[ei];
    Dual<12>dq[12];for(int d=0;d<6;++d){dq[d]=Dual<12>::variable(q[6*e.i+d],d);dq[6+d]=Dual<12>::variable(q[6*e.j+d],6+d);}
    Dual<12>strain[6];rod_strains(dq,nodes[e.i],nodes[e.j],e,strain);
    const double area=kPi*material.radius*material.radius,inertia=kPi*material.radius*material.radius*material.radius*material.radius/4.0;
    const double shear_modulus=material.young_modulus/(2.0*(1.0+material.poisson_ratio));
    const double weights[6]{material.shear_correction*shear_modulus*area,material.shear_correction*shear_modulus*area,
        material.young_modulus*area,material.young_modulus*inertia,material.young_modulus*inertia,2.0*shear_modulus*inertia};
    Dual<12>re(0.0);for(int s=0;s<6;++s)re=re+Dual<12>(0.5*e.rest_length*weights[s])*strain[s]*strain[s];
    atomicAdd(&stats->objective,re.v);atomicAdd(&stats->elastic,re.v);
    if(gradient){if(!nodes[e.i].fixed)for(int d=0;d<6;++d)atomicAdd(&gradient[6*e.i+d],re.d[d]);
        if(!nodes[e.j].fixed)for(int d=0;d<6;++d)atomicAdd(&gradient[6*e.j+d],re.d[6+d]);}
    if(diagonal){for(int a=0;a<12;++a)for(int b=0;b<12;++b){double h=0.0;for(int s=0;s<6;++s)h+=e.rest_length*weights[s]*strain[s].d[a]*strain[s].d[b];
        if(a<6&&b<6&&!nodes[e.i].fixed)atomicAdd(&diagonal[36*e.i+6*a+b],h);
        else if(a>=6&&b>=6&&!nodes[e.j].fixed)atomicAdd(&diagonal[36*e.j+6*(a-6)+(b-6)],h);
        else if(a<6&&b>=6&&!nodes[e.i].fixed&&!nodes[e.j].fixed)upper[36*e.i+6*a+(b-6)]+=h;}}
    if(!has_collider||nodes[e.i].fixed||!e.collider_contact)return;const DVec3 p0=make_vec(q[6*e.i],q[6*e.i+1],q[6*e.i+2]),p1=make_vec(q[6*e.j],q[6*e.j+1],q[6*e.j+2]);
    const DVec3 lo=min_vec(p0,p1),hi=max_vec(p0,p1);const double padding=material.radius+material.collider_offset+material.barrier_distance;
    uint32_t closest_index=0xffffffffu;ClosestPair pair{1.0e300,0.0,make_vec(1,0,0)};
    query_bvh(bvh,order,lo,hi,padding,[&](uint32_t ti){atomicAdd(&stats->candidates,1ULL);const DTriangle&t=triangles[ti];
        const ClosestPair candidate=closest_segment_triangle(p0,p1,collider[t.i0],collider[t.i1],collider[t.i2]);
        if(candidate.distance<pair.distance){pair=candidate;closest_index=ti;}});
    if(closest_index==0xffffffffu)return;const DTriangle&t=triangles[closest_index];const double gap=pair.distance-material.radius-material.collider_offset;
    atomic_min_double(&stats->minimum_gap,gap);if(gap<=desc.minimum_gap){atomicExch(&stats->feasible,0);return;}if(gap>=material.barrier_distance)return;
    atomicAdd(&stats->active,1ULL);Dual<6>cq[6];for(int d=0;d<3;++d){cq[d]=Dual<6>::variable(q[6*e.i+d],d);cq[3+d]=Dual<6>::variable(q[6*e.j+d],3+d);}
    const DVec3 op0=make_vec(old_q[6*e.i],old_q[6*e.i+1],old_q[6*e.i+2]),op1=make_vec(old_q[6*e.j],old_q[6*e.j+1],old_q[6*e.j+2]);
    const double contact_stiffness=effective_contact_stiffness(nodes,e,pair.t,material,gap,dt);
    const Dual<6>ce=contact_energy_dual(cq,op0,op1,collider[t.i0],collider[t.i1],collider[t.i2],old_collider[t.i0],old_collider[t.i1],old_collider[t.i2],pair,material,contact_stiffness);
    if(!isfinite(ce.v)){atomicExch(&stats->feasible,0);return;}atomicAdd(&stats->objective,ce.v);atomicAdd(&stats->contact,ce.v);
    if(gradient){for(int d=0;d<3;++d)atomicAdd(&gradient[6*e.i+d],ce.d[d]);for(int d=0;d<3;++d)atomicAdd(&gradient[6*e.j+d],ce.d[3+d]);}
    if(diagonal){const DVec3 bp=collider[t.i0]*pair.bary.x+collider[t.i1]*pair.bary.y+collider[t.i2]*pair.bary.z;
        const DVec3 normal=(p0*(1.0-pair.t)+p1*pair.t-bp)/fmax(pair.distance,1e-15);const double aa=material.barrier_distance-gap;
        const double curvature=contact_stiffness*(-2.0*log(gap/material.barrier_distance)+4.0*aa/gap+aa*aa/(gap*gap));
        double dg[6]{normal.x*(1.0-pair.t),normal.y*(1.0-pair.t),normal.z*(1.0-pair.t),normal.x*pair.t,normal.y*pair.t,normal.z*pair.t};
        for(int a=0;a<6;++a)for(int b=0;b<6;++b){const double h=curvature*dg[a]*dg[b];if(a<3&&b<3)atomicAdd(&diagonal[36*e.i+6*a+b],h);
            else if(a>=3&&b>=3)atomicAdd(&diagonal[36*e.j+6*(a-3)+(b-3)],h);else if(a<3&&b>=3)upper[36*e.i+6*a+(b-3)]+=h;}}
}

__global__ void friction_impulse_kernel(const double*q,double*delta_velocity,const double*velocity,const DNode*nodes,
                                        const DElement*elements,uint32_t element_count,const DVec3*collider,const DVec3*old_collider,
                                        const DTriangle*triangles,const DBvhNode*bvh,const uint32_t*order,DMaterial material,double dt)
{
    const uint32_t ei=blockIdx.x*blockDim.x+threadIdx.x;if(ei>=element_count||!(material.friction>0.0))return;const DElement&e=elements[ei];if(!e.collider_contact)return;
    if(nodes[e.i].fixed&&nodes[e.j].fixed)return;const DVec3 p0=make_vec(q[6*e.i],q[6*e.i+1],q[6*e.i+2]),p1=make_vec(q[6*e.j],q[6*e.j+1],q[6*e.j+2]);
    const double padding=material.radius+material.collider_offset+material.barrier_distance;uint32_t closest_index=0xffffffffu;ClosestPair pair{1.0e300,0.0,make_vec(1,0,0)};
    query_bvh(bvh,order,min_vec(p0,p1),max_vec(p0,p1),padding,[&](uint32_t ti){const DTriangle&t=triangles[ti];
        const ClosestPair candidate=closest_segment_triangle(p0,p1,collider[t.i0],collider[t.i1],collider[t.i2]);if(candidate.distance<pair.distance){pair=candidate;closest_index=ti;}});
    if(closest_index==0xffffffffu)return;const double gap=pair.distance-material.radius-material.collider_offset;if(!(gap>0.0&&gap<material.barrier_distance))return;
    const DTriangle&t=triangles[closest_index];const double w0=1.0-pair.t,w1=pair.t;
    const DVec3 hair_point=p0*w0+p1*w1,body_point=collider[t.i0]*pair.bary.x+collider[t.i1]*pair.bary.y+collider[t.i2]*pair.bary.z;
    const DVec3 old_body_point=old_collider[t.i0]*pair.bary.x+old_collider[t.i1]*pair.bary.y+old_collider[t.i2]*pair.bary.z;
    const DVec3 normal=(hair_point-body_point)/fmax(pair.distance,1.0e-15);
    const DVec3 v0=make_vec(velocity[6*e.i],velocity[6*e.i+1],velocity[6*e.i+2]),v1=make_vec(velocity[6*e.j],velocity[6*e.j+1],velocity[6*e.j+2]);
    const DVec3 relative=v0*w0+v1*w1-(body_point-old_body_point)/dt,tangent=relative-normal*dot(normal,relative);const double tangent_speed=norm(tangent);
    if(tangent_speed<1.0e-15)return;double inverse_mass=0.0;if(!nodes[e.i].fixed)inverse_mass+=w0*w0/fmax(nodes[e.i].mass,1.0e-30);if(!nodes[e.j].fixed)inverse_mass+=w1*w1/fmax(nodes[e.j].mass,1.0e-30);
    if(!(inverse_mass>0.0))return;const double aa=material.barrier_distance-gap,lg=log(gap/material.barrier_distance);
    const double contact_stiffness=effective_contact_stiffness(nodes,e,pair.t,material,gap,dt);
    const double normal_force=contact_stiffness*(aa*aa/gap-2.0*aa*lg),required_impulse=tangent_speed/inverse_mass;
    const double impulse_magnitude=fmin(required_impulse,material.friction*normal_force*dt),smooth_speed=material.friction_smoothing/dt;
    const DVec3 impulse=tangent*(-impulse_magnitude/fmax(sqrt(tangent_speed*tangent_speed+smooth_speed*smooth_speed),1.0e-30));
    if(!nodes[e.i].fixed){const double scale=w0/fmax(nodes[e.i].mass,1.0e-30);atomicAdd(&delta_velocity[6*e.i],impulse.x*scale);atomicAdd(&delta_velocity[6*e.i+1],impulse.y*scale);atomicAdd(&delta_velocity[6*e.i+2],impulse.z*scale);}
    if(!nodes[e.j].fixed){const double scale=w1/fmax(nodes[e.j].mass,1.0e-30);atomicAdd(&delta_velocity[6*e.j],impulse.x*scale);atomicAdd(&delta_velocity[6*e.j+1],impulse.y*scale);atomicAdd(&delta_velocity[6*e.j+2],impulse.z*scale);}
}

__device__ void record_sweep_failure(DMovingSweepFailure*failure,const DElement&e,uint32_t ei,uint32_t ti,
                                     DVec3 p0,DVec3 p1,const ClosestPair&pair,DVec3 a,DVec3 b,DVec3 c,
                                     double target,double speed)
{
    if(atomicCAS(&failure->claimed,0,1)!=0)return;failure->strand_index=e.strand;failure->element_index=ei;failure->triangle_index=ti;
    failure->distance=pair.distance;failure->required_distance=target;failure->clearance=pair.distance-target;failure->collider_displacement=speed;
    failure->hair_start=p0;failure->hair_end=p1;failure->collider_point=a*pair.bary.x+b*pair.bary.y+c*pair.bary.z;
}

__global__ void moving_sweep_kernel(const double*q,const DNode*nodes,const DElement*elements,uint32_t element_count,
                                    const DVec3*old_collider,const DVec3*collider,const DTriangle*triangles,
                                    const DBvhNode*bvh,const uint32_t*order,DMaterial material,DDesc desc,double*limit,DMovingSweepFailure*failure)
{
    const uint32_t ei=blockIdx.x*blockDim.x+threadIdx.x;if(ei>=element_count)return;const DElement&e=elements[ei];if(nodes[e.i].fixed||!e.collider_contact)return;
    const DVec3 p0=make_vec(q[6*e.i],q[6*e.i+1],q[6*e.i+2]),p1=make_vec(q[6*e.j],q[6*e.j+1],q[6*e.j+2]);
    const double target=material.radius+material.collider_offset+desc.minimum_gap;
    const double maximum_parking_clearance=fmax(4.0*desc.minimum_gap,0.25*material.barrier_distance);
    query_bvh(bvh,order,min_vec(p0,p1),max_vec(p0,p1),target,[&](uint32_t ti){
        const DTriangle&t=triangles[ti];const DVec3 da=collider[t.i0]-old_collider[t.i0],db=collider[t.i1]-old_collider[t.i1],dc=collider[t.i2]-old_collider[t.i2];
        const double speed=fmax(norm(da),fmax(norm(db),norm(dc)));if(speed<1e-30)return;double alpha=0,initial_clearance=0;bool end=false;ClosestPair pair{};DVec3 a{},b{},c{};
        for(int k=0;k<80;++k){a=old_collider[t.i0]+da*alpha;b=old_collider[t.i1]+db*alpha;c=old_collider[t.i2]+dc*alpha;pair=closest_segment_triangle(p0,p1,a,b,c);
            // Preserve a fraction of the clearance that actually exists at the
            // beginning of this proposal. A fixed parking distance can exceed
            // an already-active contact gap and incorrectly collapse TOI to zero.
            const double clearance=pair.distance-target;if(k==0)initial_clearance=fmax(0.0,clearance);const double safe_clearance=fmin(maximum_parking_clearance,0.1*initial_clearance);
            if(clearance<=0){record_sweep_failure(failure,e,ei,ti,p0,p1,pair,a,b,c,target,speed);atomic_min_double(limit,fmax(0.0,alpha-safe_clearance/speed));return;}const double advance=0.8*clearance/speed;if(alpha+advance>=1){end=true;break;}if(advance<1e-12){record_sweep_failure(failure,e,ei,ti,p0,p1,pair,a,b,c,target,speed);atomic_min_double(limit,fmax(0.0,alpha-safe_clearance/speed));return;}alpha+=advance;}
        if(!end){const double safe_clearance=fmin(maximum_parking_clearance,0.1*initial_clearance);record_sweep_failure(failure,e,ei,ti,p0,p1,pair,a,b,c,target,speed);atomic_min_double(limit,fmax(0.0,alpha-safe_clearance/speed));}});
}

__global__ void ccd_limit_kernel(const double*q,const double*direction,const DNode*nodes,const DElement*elements,
                                 uint32_t element_count,const DVec3*collider,const DTriangle*triangles,
                                 const DBvhNode*bvh,const uint32_t*order,DMaterial material,DDesc desc,double*limit)
{
    const uint32_t ei=blockIdx.x*blockDim.x+threadIdx.x;if(ei>=element_count)return;const DElement&e=elements[ei];if(nodes[e.i].fixed||!e.collider_contact)return;
    const DVec3 p0=make_vec(q[6*e.i],q[6*e.i+1],q[6*e.i+2]),p1=make_vec(q[6*e.j],q[6*e.j+1],q[6*e.j+2]);
    const DVec3 d0=make_vec(direction[6*e.i],direction[6*e.i+1],direction[6*e.i+2]),d1=make_vec(direction[6*e.j],direction[6*e.j+1],direction[6*e.j+2]);
    DVec3 lo=min_vec(min_vec(p0,p1),min_vec(p0+d0,p1+d1)),hi=max_vec(max_vec(p0,p1),max_vec(p0+d0,p1+d1));
    const double target=material.radius+material.collider_offset+desc.minimum_gap;query_bvh(bvh,order,lo,hi,target,[&](uint32_t ti){const DTriangle&t=triangles[ti];
        const double speed=fmax(norm(d0),norm(d1));if(speed<1e-30)return;double alpha=0;for(int k=0;k<80;++k){const double dist=closest_segment_triangle(p0+d0*alpha,p1+d1*alpha,collider[t.i0],collider[t.i1],collider[t.i2]).distance;
            const double clear=dist-target;if(clear<=0){atomic_min_double(limit,fmax(0.0,alpha*0.9));return;}const double advance=0.8*clear/speed;if(alpha+advance>=1)return;if(advance<1e-12){atomic_min_double(limit,alpha);return;}alpha+=advance;}atomic_min_double(limit,alpha);});
}

__global__ void gather_original_kernel(const double*q,const uint32_t*mapping,uint32_t count,DVec3*out)
{
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){const uint32_t n=mapping[i];out[i]=make_vec(q[6*n],q[6*n+1],q[6*n+2]);}
}
__global__ void gather_internal_kernel(const double*q,uint32_t count,DVec3*out)
{
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)out[i]=make_vec(q[6*i],q[6*i+1],q[6*i+2]);
}

} // namespace
} // namespace kami
