/* Copyright 2012-2014 The Pennsylvania State University
 *
 * This software was written by David Hadka and others.
 * 
 * The use, modification and distribution of this software is governed by the
 * The Pennsylvania State University Research and Educational Use License.
 * You should have received a copy of this license along with this program.
 * If not, contact <info@borgmoea.org>.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include "borgms.h"

#define PI 3.14159265358979323846

int nvars = 11;
int nobjs = 2;

// The evaluation function is defined in the same way as the serial example
// (see dtlz2_serial.c), but is evaluated in parallel by the slave processes.
void dtlz2(double* vars, double* objs, double* consts) {
	int i;
	int j;
	int k = nvars - nobjs + 1;
	double g = 0.0;

	for (i=nvars-k; i<nvars; i++) {
		g += pow(vars[i] - 0.5, 2.0);
	}

	for (i=0; i<nobjs; i++) {
		objs[i] = 1.0 + g;

		for (j=0; j<nobjs-i-1; j++) {
			objs[i] *= cos(0.5*PI*vars[j]);
		}

		if (i != 0) {
			objs[i] *= sin(0.5*PI*vars[nobjs-i-1]);
		}
	}
}

int main(int argc, char* argv[]) {
	int i, j;
	int rank;
	char runtime[256];

	// All master-slave runs need to call startup and set the execution
	// limit based on the number of function evaluations.
	BORG_Algorithm_ms_startup(&argc, &argv);
	BORG_Algorithm_ms_max_evaluations(100000);

	// Alternatively, we can terminate after a fixed wallclock time,
	// given in hours.
	//BORG_Algorithm_ms_max_time(0.1);

	// Define the problem.  Problems are defined the same way as the
	// serial example (see dtlz2_serial.c).
	BORG_Problem problem = BORG_Problem_create(nvars, nobjs, 0, dtlz2);

	for (j=0; j<nvars; j++) {
		BORG_Problem_set_bounds(problem, j, 0.0, 1.0);
	}

	for (j=0; j<nobjs; j++) {
		BORG_Problem_set_epsilon(problem, j, 0.01);
	}

	// Get the rank of this process.  The rank is used to ensure each
	// parallel process uses a different random seed.
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// When running experiments, we want to run the algorithm multiple
	// times and average the results.
	for (i=0; i<10; i++) {
		// Save runtime dynamics to a file.  Only the master node
		// will write to this file.  Note how we create separate
		// files for each run.
		sprintf(runtime, "runtime_%d.txt", i);
		BORG_Algorithm_output_runtime(runtime);

		// Seed the random number generator.
		BORG_Random_seed(37*i*(rank+1));

		// Run the master-slave Borg MOEA on the problem.
		BORG_Archive result = BORG_Algorithm_ms_run(problem);

		// Only the master process will return a non-NULL result.
		// Print the Pareto optimal solutions to the screen.
		if (result != NULL) {
			printf("Seed %d:\n", i);
			BORG_Archive_print(result, stdout);
			BORG_Archive_destroy(result);
			printf("\n");
		}
	}

	// Shutdown the parallel processes and exit.
	BORG_Algorithm_ms_shutdown();
	BORG_Problem_destroy(problem);
	return EXIT_SUCCESS;
}
