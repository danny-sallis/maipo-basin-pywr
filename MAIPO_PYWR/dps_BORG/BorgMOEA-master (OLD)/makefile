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
	$(CC) $(CFLAGS) -o dtlz2_serial.exe dtlz2_serial.c borg.c mt19937ar.c $(LIBS)
	$(CC) $(CFLAGS) -o dtlz2_advanced.exe dtlz2_advanced.c borg.c mt19937ar.c $(LIBS)
	$(CC) $(CFLAGS) -o borg.exe frontend.c borg.c mt19937ar.c $(LIBS)

compile-parallel:
	$(MPICC) $(CFLAGS) -o dtlz2_ms.exe dtlz2_ms.c borgms.c mt19937ar.c $(LIBS)

.PHONY: compile, compile-parallel

