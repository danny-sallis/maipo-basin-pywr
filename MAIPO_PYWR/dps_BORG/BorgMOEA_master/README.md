# Borg MOEA
The original implementation of the Borg MOEA in the C programming language. This is a high-performance
implementation that supports both serial and master-slave parallelization using MPI.

## Prerequisites
A C/C++ compiler and Make are required to build and run this code.  The instructions below use the GNU C/C++ 
compiler (gcc), but any compiler should work.  Windows users may use [MinGW](https://www.mingw-w64.org/),
[Cygwin](https://www.cygwin.com/), or the Visual Studio C++ compiler.

MPI is required to use the parallel version.

## Examples
To compile and run the serial examples:

```bash

make                    # Compiles the Borg MOEA and examples

./dtlz2_serial.exe      # Demonstrates solving the DTLZ2 problem using the serial implementation
./dtlz2_advanced.exe    # Advanced example, demonstrating custom parameters, saving results to file, and checkpoints
```

A command-line tool is also provided that let's us solve optimization problems defined in a separate program.
Here, we solve the DTLZ2 problem written in Python:

```bash
# Display usage information
./borg.exe -h

# Solve the DTLZ2 problem with 11 decision variables and two objectives
./borg.exe -n 10000 -v 11 -o 2 -e 0.01,0.01 python dtlz2.py

# Solve the same DTLZ2 problem, but specify the lower and upper bounds for decision variables
./borg.exe -n 10000 -v 11 -o 2 -e 0.01,0.01 -l 0,0,0,0,0,0,0,0,0,0,0 -u 1,1,1,1,1,1,1,1,1,1,1 python dtlz2.py
```

Lastly, we provide [plugins for popular programming languages](plugins), including Java, C#, Python, and R.  Instructions
for using these plugins are provided alongside each.

## Parallel Example

This code also supports master-slave parallelization using MPI.  An example is provided in
`dtlz2_ms.c` which can be compiled using:

```bash

make compile-parallel
```

For example, here we use the Local Area Multicomputer (LAM) engine to run the MPI program
locally:

```bash

sudo apt install -y mpi mpich libmpich-dev libopenmpi-dev
make compile-parallel
lamboot
mpirun -np 4 ./dtlz2_ms.exe
lamhalt
```

On larger shared systems, the exact steps to run MPI programs vary.  For example,
if using the Portable Batch System (PBS), we would create a PBS file named `dtlz2_ms.pbs`
containing:

```
#PBS -l nodes=16:ppn=4
#PBS -l walltime=96:00:00
#PBS -j oe
#PBS -o output.txt
cd $PBS_O_WORKDIR
module load openmpi/gnu
mpirun dtlz2_ms.exe
```

and submit the job using:

```bash

qsub dtlz2_ms.pbs
```

## Citations

> Hadka, D., and Reed, P.M. "Borg: An Auto-Adaptive Many-Objective Evolutionary Computing Framework." Evolutionary Computation, 21(2):231-259, 2013.

> Hadka, D., and Reed, P. "Large-scale Parallelization of the Borg MOEA to Enhance the Management of Complex Environmental Systems." Environmental Modelling & Software, doi:10.1016/j.envsoft.2014.10.014, 2014.

> Reed, P. and Hadka, D.  "Evolving Many-Objective Water Management to Exploit Exascale Computing."  Water Resources Research, 50(10):7692-7713, 2014.

## License
Copyright 2012-2014 The Pennsylvania State University, Copyright 2018-2023 David Hadka.

The use, modification and distribution of this software is governed by The Pennsylvania State University Research and Educational Use License.
You should have received a copy of this license along with this program. If not, contact info@borgmoea.org.

These codes use the Mersenne Twister pseudo-random number generator by Takuji Nishimura and Makoto Matsumoto.  These codes are licensed under the
Modified BSD license.  See the copyright and license notices in mt19937ar.c for details.

