PROG =	hde pmds

SRCS =	hde.c pivot_mds.c main.c

OBJS =	hde.o pivot_mds.o main.o

GVIZ_BUILD_PATH = /home/yifanhu/GraphViz/graphviz2/build

ifeq ($(shell uname -s),Darwin)
GVIZ_BUILD_PATH = /Users/yifanhu/GraphViz/graphviz_head/build
endif

ifeq ($(HOST),myverse.research.att.com)
GVIZ_BUILD_PATH = /fs/myverse/RECOMMENDERS/codes/yifanhu/GraphViz/graphviz/build
endif

LIBS = -L$(GVIZ_BUILD_PATH)/lib -lcgraph -lcdt -lm -L../sfdp -lsfdp	

CC = cc
CFLAGS = -O3 -DWITH_CGRAPH -DHAVE_DOT -I../sfdp -I$(GVIZ_BUILD_PATH)/include/graphviz/ -Wall -DTIME
LDFLAGS = 

all: $(PROG)

$(PROG): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)
	ln -sf hde pmds

clean:
	rm -f $(PROG) $(OBJS) *.mod

.SUFFIXES: $(SUFFIXES) .f90

.f90.o:
	$(F90) $(F90FLAGS) -c $<

hde.o: hde.h
main.o: hde.h 
pivot_mds: hde.h

