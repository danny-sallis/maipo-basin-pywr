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
#include <signal.h>
#include <string.h>
#include <errno.h>
#include "borg.h"

#define PI 3.14159265358979323846

int nvars = 11;
int nobjs = 2;

// Define the objective function.
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
	int i;

	// Setup parameters for the run.
	int maxEvaluations = 1000000;
	const char* outputFile = "output.txt";
	int checkpointFrequency = 10000;
	const char* checkpointFile = "checkpoint.txt";

	// Set the random number seed generator.
	BORG_Random_seed(1337);

	// Temporary file for writing checkpoint data to avoid corrupting the
	// previous checkpoint in case an error or interrupt occurs.
	const char* checkpointSwapFile = ".checkpoint.tmp"; 

	// Define the optimization problem.
	BORG_Problem problem = BORG_Problem_create(nvars, nobjs, 0, dtlz2);

	for (i=0; i<nvars; i++) {
		BORG_Problem_set_bounds(problem, i, 0.0, 1.0);
	}

	for (i=0; i<nobjs; i++) {
		BORG_Problem_set_epsilon(problem, i, 0.01);
	}

	// Configure each of the operators.
	BORG_Operator pm = BORG_Operator_create("PM", 1, 1, 2, BORG_Operator_PM);
	BORG_Operator_set_parameter(pm, 0, 1.0 / BORG_Problem_number_of_variables(problem));
	BORG_Operator_set_parameter(pm, 1, 20.0);

	BORG_Operator sbx = BORG_Operator_create("SBX", 2, 2, 2, BORG_Operator_SBX);
	BORG_Operator_set_parameter(sbx, 0, 1.0);
	BORG_Operator_set_parameter(sbx, 1, 15.0);
	BORG_Operator_set_mutation(sbx, pm);

	BORG_Operator de = BORG_Operator_create("DE", 4, 1, 2, BORG_Operator_DE);
	BORG_Operator_set_parameter(de, 0, 0.1);
	BORG_Operator_set_parameter(de, 1, 0.5);
	BORG_Operator_set_mutation(de, pm);

	BORG_Operator um = BORG_Operator_create("UM", 1, 1, 1, BORG_Operator_UM);
	BORG_Operator_set_parameter(um, 0, 1.0 / BORG_Problem_number_of_variables(problem));

	BORG_Operator spx = BORG_Operator_create("SPX", 10, 2, 1, BORG_Operator_SPX);
	BORG_Operator_set_parameter(spx, 0, 3.0);

	BORG_Operator pcx = BORG_Operator_create("PCX", 10, 2, 2, BORG_Operator_PCX);
	BORG_Operator_set_parameter(pcx, 0, 0.1);
	BORG_Operator_set_parameter(pcx, 1, 0.1);

	BORG_Operator undx = BORG_Operator_create("UNDX", 10, 2, 2, BORG_Operator_UNDX);
	BORG_Operator_set_parameter(undx, 0, 0.5);
	BORG_Operator_set_parameter(undx, 1, 0.35);

	// Setup the algorithm and specify the operators.
	BORG_Algorithm algorithm = BORG_Algorithm_create(problem, 6);
	BORG_Algorithm_set_operator(algorithm, 0, sbx);
	BORG_Algorithm_set_operator(algorithm, 1, de);
	BORG_Algorithm_set_operator(algorithm, 2, pcx);
	BORG_Algorithm_set_operator(algorithm, 3, spx);
	BORG_Algorithm_set_operator(algorithm, 4, undx);
	BORG_Algorithm_set_operator(algorithm, 5, um);

	// Load and resume from the checkpoint if the file exists.
	void (*signal_handler)(int);
	FILE* checkpoint = fopen(checkpointFile, "r");
	int lastCheckpoint = 0;
			
	if (checkpoint != NULL) {
		printf("Resuming from checkpoint file!\n");
		BORG_Algorithm_restore(algorithm, checkpoint);
		fclose(checkpoint);
		checkpoint = NULL;
		lastCheckpoint = BORG_Algorithm_get_nfe(algorithm);
	}

	// Run each iteration of the optimization algorithm.
	while (BORG_Algorithm_get_nfe(algorithm) < maxEvaluations) {
		BORG_Algorithm_step(algorithm);

		// Periodically save the checkpoint file.
		if (BORG_Algorithm_get_nfe(algorithm) >= lastCheckpoint+checkpointFrequency) {
			checkpoint = fopen(checkpointSwapFile, "w");
			
			if (checkpoint != NULL) {
				BORG_Algorithm_checkpoint(algorithm, checkpoint);
				fclose(checkpoint);
				checkpoint = NULL;

				// Swap the temporary checkpoint file for the actual checkpoint file
				signal_handler = signal(SIGINT, SIG_IGN);
				remove(checkpointFile);
				
				if (rename(checkpointSwapFile, checkpointFile) != 0) {
					printf("Unable to rename %s to %s: %s\n", checkpointSwapFile, checkpointFile, strerror(errno));
				}

				signal(SIGINT, signal_handler);
			} else {
				printf("Unable to save checkpoint file to %s: %s\n", checkpointSwapFile, strerror(errno));
			}

			lastCheckpoint = BORG_Algorithm_get_nfe(algorithm);
		}
	}

	// Save the Pareto optimal solutions to a file.
	BORG_Archive result = BORG_Algorithm_get_result(algorithm);
	FILE* output = fopen(outputFile, "w");
	BORG_Validate_file(output);
	BORG_Archive_print(result, output);
	fclose(output);
	printf("Output written to %s\n", outputFile);

	// Delete the checkpoint file after the result was successfully written.
	remove(checkpointFile);

	// Free any allocated memory.
	BORG_Operator_destroy(sbx);
	BORG_Operator_destroy(de);
	BORG_Operator_destroy(pm);
	BORG_Operator_destroy(um);
	BORG_Operator_destroy(spx);
	BORG_Operator_destroy(pcx);
	BORG_Operator_destroy(undx);
	BORG_Algorithm_destroy(algorithm);
	BORG_Archive_destroy(result);
	BORG_Problem_destroy(problem);

	return EXIT_SUCCESS;
}

