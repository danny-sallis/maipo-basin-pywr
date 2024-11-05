CC = gcc
MPICC = mpicc
CFLAGS = -O3
LDFLAGS = -Wl,-R,\.
LIBS = -lm
UNAME_S = $(shell uname -s)

ifneq (, $(findstring SunOS, $(UNAME_S)))
	LIBS += -lnsl -lsocket -lresolv
endif

compile:
	$(CC) $(CFLAGS) $(LDFLAGS) -o dtlz2_serial.exe dtlz2_serial.c borg.c mt19937ar.c $(LIBS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o dtlz2_advanced.exe dtlz2_advanced.c borg.c mt19937ar.c $(LIBS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o borg.exe frontend.c borg.c mt19937ar.c $(LIBS)

compile-parallel:
	$(MPICC) $(CFLAGS) -o dtlz2_ms.exe dtlz2_ms.c borgms.c mt19937ar.c $(LIBS)
	$(MPICC) $(CFLAGS) -shared -fPIC -o libborgms.so borgms.c mt19937ar.c $(LIBS)

ifneq ("$(wildcard borgmm.*)","")
	$(MPICC) $(CFLAGS) -o dtlz2_mm.exe dtlz2_mm.c borgmm.c mt19937ar.c $(LIBS)
	$(MPICC) $(CFLAGS) -shared -fPIC -o libborgmm.so borgmm.c mt19937ar.c $(LIBS)
endif

.PHONY: compile, compile-parallel
