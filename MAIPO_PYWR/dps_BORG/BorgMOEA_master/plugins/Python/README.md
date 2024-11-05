# Python Plugin

Follow the [Compiling the Shared Library](..#compiling-the-shared-library) instructions to compile the
Borg MOEA shared library.  Then, copy the library into this folder.  Finally, run:

```bash

python example.py
```

## Master-Slave Parallelization

The Python plugin also supports master-slave parallelization with the `mpitest.py` example.  First,
compile the master-slave shared library and copy it into this directory:

```bash

mpicc -shared -fPIC -O3 -o libborgms.so borgms.c mt19937ar.c -lm
```

To test locally, we can use the Local Area Multicomputer (LAM) engine to run the MPI program:

```bash

lamboot
mpirun -np 2 python mpi_example.py
lamhalt
```

Or, for example, submit a job using the Portable Batch System (PBS).  The specifics
will depend on the system being used.

```
#PBS -l nodes=1:ppn=4
#PBS -l walltime=96:00:00
#PBS -j oe
#PBS -o mpitest.out

cd $PBS_O_WORKDIR

module load openmpi/gnu
module load python/3.10.4
mpirun python mpi_example.py
```
