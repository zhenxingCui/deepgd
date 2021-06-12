#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include "mmio.h"
#include "SparseMatrix.h"
#include "matrix_market.h"
#include "QuadTree.h"
#include "spring_electrical.h"
#include "Multilevel.h"
#include <time.h>
#include "post_process.h"
#include "uniform_stress.h"
#include "stress_model.h"
#include "get_ps.h"
#include "hde.h"
#include "pivot_mds.h"
#include "DotIO.h"
#include "general.h"

#define FMT_NONE 0
#define FMT_MATHEMATICA 1
#define FMT_DOT 2
#define FMT_COORDINATES 3
#define FMT_PS 4
int format = FMT_DOT;
int Verbose;
double scale = 0;


static FILE* openF (char* fname, char* mode)
{
    FILE* f = fopen (fname, mode);
    if (!f) {
        fprintf (stderr, "Could not open %s for %s\n", fname,
            ((*mode == 'r') ? "reading" : "writing"));
        exit (1);
    }
    return f;
}

static void usage (char* cmd, int eval)
{
    fprintf(stderr, "Usage: %s <options> filename\n", cmd);
    fprintf(stderr, "Options are\n");
    fprintf(stderr, "-k K           number of pivot nodes\n");
    fprintf(stderr, "-u             whether the graph is assumed to be undriected with no loops. Default is directed.\n");
    fprintf(stderr, "-centers=i1,i2,...ik    manually gives the k-centers. The number of centers must be >= the value of -k option.\n");
    fprintf(stderr, "-connected             whether to make disconnected graphs conected by adding a minimum number of edges\n");
    fprintf(stderr, "-d             dimension to generate layout. 2 (default) or 3.\n");
    fprintf(stderr, "-v             verbose output.\n");
    fprintf(stderr, "-o file        output file name\n");
    fprintf(stderr, "-s x           scale output in dot by x. If 0, no scaling, If > 0, scale to a fraction of 7x10 paper size, if x < 0, scale coordinates by -x. Default is x = 0.\n");
    fprintf(stderr, "-T fmt         set output format: m(athematica) d(ot) c(oordinate), p(postscript)\n");
    fprintf(stderr, "-U i           whether to treat unsymmetric square matrix as a bipartite graph. 0 (never), 1 (default, pattern unsymmetric as bipartite), 2 (unsymmetric matrix always as bipartite. 3: always treat square matrix as bipartite. This option only has an effect on matrix input\n");
   exit(eval);
}

static void init(int argc, char *argv[], int *K, char **infile, char **outfile, int *dim, int *undirected, int *bipartite, int *make_connected, int **centers){

  unsigned int c;
  char *scenters;

  char* cmd = argv[0];
  
  *K = 50;
  *infile = NULL;
  *outfile = NULL;
  Verbose = FALSE;
  *dim = 2;
  *undirected = FALSE;
  *bipartite = BIPARTITE_PATTERN_UNSYM;

  while ((c = getopt(argc, argv, ":vuU:k:d:s:T:c:")) != -1) {
    switch (c) {
    case 'c':
      if (strncmp(optarg,"onnected", 8) == 0){
	*make_connected = TRUE;
      } else if (strncmp(optarg,"enters=", 7) == 0){
	scenters = MALLOC(sizeof(char)*(strlen(optarg)+1));
	strcpy(scenters, optarg); 
	*centers = MALLOC(sizeof(int)*(*K));
      } else {
	fprintf(stderr,"-c option must be of the form -centers=x, where x is a list integers separated by commas.\n");
	usage(cmd, 1);
      }
      break;
    case 'd':
      if (sscanf(optarg,"%d",dim) <= 0) {
	fprintf(stderr, "-d arg %s must be an integer\n",optarg);
	*dim = 2;
      }
      break;
    case 'u':
      *undirected = TRUE;
      break;
    case 'v':
      Verbose = TRUE;
      break;
    case 'o':
      *outfile = optarg;
      break;
    case 's':{
      real p;
      if (sscanf(optarg,"%lf",&p) <= 0) {
	fprintf(stderr, "-s arg %s must be a real number\n",optarg);
      } else {
	scale = p;
      }
      break;
    }
    case 'T':
      switch (*optarg) {
      case 'm' :
	format = FMT_MATHEMATICA;
	break;
      case 'd' :
	format = FMT_DOT;
	break;
      case 'c' :
	format = FMT_COORDINATES;
	break;
      case 'p' :
	format = FMT_PS;
	break;
      default :
	fprintf(stderr, "option -T%s unrecognized - ignored\n", optarg);
	break;
      }
      break;
    case 'k':{
      int k;
      if (sscanf(optarg,"%d",&k) <= 0 || k < 2) {
	usage(cmd, 1);
      } else {
	*K = k;
	if (*centers) *centers = REALLOC(*centers, sizeof(int)*k);      
      }
      break;
    }
    case 'U':{
      int u;
      if (sscanf(optarg,"%d",&u) <= 0 || u < 0 || u > BIPARTITE_ALWAYS) {
	usage(cmd, 1);
      } else {
	*bipartite = u;
      }
      break;
    }
    case '?':
      if (optopt == '?')
	usage(cmd, 0);
      else
	fprintf(stderr, "gc: option -%c unrecognized - ignored\n",
		optopt);
      break;
    }
  }


  if (*centers){
    char *ss = scenters;
    int i;

    assert(scenters);
    ss += 7;/* optargs = enters=1,4,2,6...*/
    for (i = 0; i < *K; i++){
      if (sscanf(ss,"%d", &((*centers)[i])) <= 0){
	fprintf(stderr,"-c%s option must be of the form -centers=x, where x is a list of %d integers separated by commas.\n", scenters, *K);
	usage(cmd, 1);
      }
      if (i < *K-1){
	ss = strstr(ss, ",");
	if (!ss){
	  fprintf(stderr,"-c%s option must be of the form -centers=x, where x is a list of %d integers separated by commas.\n", scenters, *K);
	  usage(cmd, 1);
	}
	ss++;
      }
    }
    FREE(scenters);

  }
  argv += optind;
  argc -= optind;
  
  if (argc)
    *infile = argv[0];
  
}

static int isDotFile (char* fname){
  char* dotp = strrchr (fname, '.');
  return (dotp && (!strcmp (dotp+1, "dot") || !strcmp (dotp+1, "gv")));
}



int main(int argc, char *argv[])
{
  Agraph_t* g = 0;
  int dotFile = 0;
  char *infile, *outfile;
  FILE *f;
  SparseMatrix A = NULL, D = NULL;
  int dim;
  real *x = NULL;
  int flag, undirected;
  real *label_sizes = NULL;

  //  int with_color = FALSE, with_label = FALSE,random_edge_color = FALSE, use_matrix_value = FALSE, whitebg = TRUE;
  int with_color = FALSE, with_label = FALSE,random_edge_color = FALSE, use_matrix_value = FALSE, whitebg = TRUE;


  int i;
  int n_edge_label_nodes = 0, *edge_label_nodes = NULL;

  int bipartite;
  int matrix_modified = 0;
  int flag_bipartite = 1<<0,flag_largest_comp = 1<<1,flag_not_square = 1<<2, flag_make_weakly_connected = 1<<3;
  int K;
  enum {METHOD_HDE, METHOD_PIVOT_MDS};
  int method = METHOD_HDE;
  int *centers = NULL;
  int make_connected = FALSE;

#ifdef TIME
  clock_t start;
#endif

  init(argc, argv, &K, &infile, &outfile, &dim, &undirected, &bipartite, &make_connected, &centers);

  if (strcmp(strip_dir(argv[0]),"pmds") == 0) method = METHOD_PIVOT_MDS;

  if (infile) {
    dotFile = isDotFile (infile);
    f = openF (infile, "r");
  } else {
    f = stdin;
  }

  if (dotFile) {
#ifdef WITH_CGRAPH
    g = agread (f, 0);
    aginit(g, AGNODE, "nodeinfo", sizeof(Agnodeinfo_t), TRUE);
#else
    aginit ();
    g = agread (f);
#endif
    A = SparseMatrix_import_dot(g, dim, &label_sizes, &x, &n_edge_label_nodes, &edge_label_nodes, FORMAT_CSR, &D);
  } else {
    A = SparseMatrix_import_matrix_market(f, FORMAT_CSR);
    if (A) {
      if (A->m != A->n) set_flag(matrix_modified,flag_not_square);
      SparseMatrix B = SparseMatrix_to_square_matrix(A, bipartite);
      if (A!=B) set_flag(matrix_modified,flag_bipartite);
      A = B;
      D = SparseMatrix_copy(A);
      D = SparseMatrix_set_entries_to_real_one(D);
   }
  }

  if (!A) {
    if (f == stdin){
#ifdef WITH_CGRAPH
      g = agread (f, 0);
      aginit(g, AGNODE, "nodeinfo", sizeof(Agnodeinfo_t), TRUE);
#else
      aginit ();
      g = agread (f);
#endif
      A = SparseMatrix_import_dot(g, dim, &label_sizes, &x, &n_edge_label_nodes, &edge_label_nodes, FORMAT_CSR, &D);
      if (A) dotFile = TRUE;
    }
    if (!A){
      fprintf(stderr,"can not open file %s\n",infile);
      exit(1);
    }
  }

  if (infile) fclose(f);


  /* ====== layout ==========*/

  if (!SparseMatrix_connectedQ(A)) {
    if (!make_connected){
      SparseMatrix B;
      B = SparseMatrix_largest_component(A);
      set_flag(matrix_modified, flag_largest_comp);
      fprintf(stderr,"Warning: the graph with %d vertices is disconnected!! I am taking the largest component of size %d\n", A->m, B->m);
      SparseMatrix_delete(A);
      A = B;
      if (D){
	B = SparseMatrix_largest_component(D);
	set_flag(matrix_modified, flag_largest_comp);
	fprintf(stderr,"Warning: the distance graph with %d vertices is disconnected!! I am taking the largest component of size %d\n", A->m, B->m);
	SparseMatrix_delete(D);
	D = B;
      }
    } else {
      SparseMatrix B;
      B = SparseMatrix_make_weakly_connected(A);
      set_flag(matrix_modified, flag_make_weakly_connected);
      fprintf(stderr,"Warning: the graph with %d vertices is disconnected!! I am making it connected by adding %d edges\n", A->m, B->nz - A->nz);
      SparseMatrix_delete(A);
      A = B;
      if (D){
        B = SparseMatrix_make_weakly_connected(D);
        set_flag(matrix_modified, flag_make_weakly_connected);
        fprintf(stderr,"Warning: the graph with %d vertices is disconnected!! I am making it connected by adding %d edges\n", D->m, B->nz - D->nz);
        SparseMatrix_delete(D);
        D = B;
      }
    }
  }
  if (!x) {
    x = MALLOC(sizeof(real)*A->m*dim);
    srand(123);
    for (i = 0; i < dim*A->m; i++) x[i] = 72*drand();
  }

#ifdef TIME
  start = clock();
#endif

  D = SparseMatrix_symmetrize_nodiag(D, FALSE);

  if (centers){
    for (i = 0; i < K; i++){
      if (centers[i] < 0 || centers[i] >= D->m){
	fprintf(stderr,"-centers= values must be between 0 and %d, value of %d is outside of that range.\n", D->m, centers[i]);
	usage("cmd",1);
      }
    }
  }
  switch (method){
  case METHOD_HDE:
    hde(dim, D, K, x, centers, &flag);
    break;
  case METHOD_PIVOT_MDS:
    pivot_mds(dim, D, K, x, centers, &flag); 
    break;
  } 


  //remove added edges
  if (test_flag(matrix_modified, flag_make_weakly_connected)){
    A = SparseMatrix_crop(A, VALUE_ADDED_EDGES);
    if (D){
      D = SparseMatrix_crop(D, VALUE_ADDED_EDGES);
    }
  }

  if (D && D->m < 10000) {
    fprintf(stderr,"final full stress = %f\n",get_full_stress(D, dim, x, WEIGHTING_SCHEME_SQR_DIST));
    dump_distance_edge_length("/tmp/plotdata.txt", D, dim, x);
  }


  assert(!flag);

#ifdef TIME
  fprintf(stderr, "statistics for %s cpu=%10.3f", strip_dir(infile), ((real) (clock() - start))/CLOCKS_PER_SEC);
  if (test_flag(matrix_modified, flag_bipartite)) fprintf(stderr,", bipartite graph");
  if (test_flag(matrix_modified, flag_largest_comp)) fprintf(stderr,", largest component");
  if (test_flag(matrix_modified, flag_not_square)) fprintf(stderr,", not-square");
  fprintf(stderr,"\n");
#endif



  if (flag) exit(1);
  assert(!flag);

  if (scale > 0) scale_to_box(0,0,7*70*scale,10*70*scale,A->m,dim,x);

#ifdef HAVE_DOT
#ifdef DEBUG_PRINT
  /*  dump_coordinates(infile, A->m, dim, x);*/
#endif
#endif


  if (undirected) {
    SparseMatrix B;
    B = SparseMatrix_make_undirected(A);
    SparseMatrix_delete(A);
    A = B;
#ifdef HAVE_DOT
    if (format == FMT_DOT){
      if (g) {
	g = makeDotGraph (A, infile, dim, x, with_color, with_label, use_matrix_value);
      } else {
	g = makeDotGraph (A, infile, dim, x, with_color, with_label, use_matrix_value);
      }
    }
#endif
  }

  if (outfile)
      f = openF (outfile,"w");
  else
      f = stdout;
  
  if (format == FMT_NONE) {
    if (dotFile) 
      format = FMT_DOT;
    else 
      format = FMT_MATHEMATICA;
  }  
  switch (format){

  case FMT_DOT:
    if (!dotFile){ 
      g = makeDotGraph (A, infile, dim, x, with_color, with_label, use_matrix_value);
      
    } 
    
    fprintf(stderr,"scale=%f\n",scale);
    if (scale >= 0){
      attach_embedding(g, dim, 1, x);
    } else {
      attach_embedding(g, dim, -scale, x);
    }
    if (random_edge_color) g=assign_random_edge_color(g);    
    agwrite (g, f);
    break;
  case FMT_MATHEMATICA:
   export_embedding(f, dim, A, x, NULL);
   break;
  case FMT_COORDINATES:{
    int i, k;
    for (i = 0; i < A->m; i++){
      for (k = 0; k < dim; k++){
	fprintf(f, "%f ",x[i*dim+k]);
      }
      fprintf(f,"\n");
    }
    break;
  }
  case FMT_PS:{
    char fname[1000];
    real linewidth = LINEWIDTH_AUTO;
    // linewidth = 0.;
    strcpy(fname, strip_dir(infile));
    if (method == METHOD_HDE){
      strcat(fname,"_hde");
    } else {
      strcat(fname,"_pmds");
    }
    fprintf(stderr,"fname=%s\n",fname);
    dump_coordinates(fname, A->m, dim, x);
    export_square_matrix_as_ps(f, A, dim, x, strip_dir(infile), use_matrix_value, FALSE, FALSE, with_color, with_label, -1., linewidth, whitebg, &flag);
    if (flag) fprintf(stderr, "export to postscript failed\n");
    break;
  }
  }
  if (outfile) fclose(f);


  SparseMatrix_delete(A);
  FREE(x);
  if (label_sizes) FREE(label_sizes);
  if (edge_label_nodes) FREE(edge_label_nodes);
  return 0;
}
