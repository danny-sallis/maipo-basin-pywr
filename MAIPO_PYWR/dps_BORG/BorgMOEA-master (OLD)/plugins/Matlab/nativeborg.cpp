/* MATLAB Interface to the Borg MOEA
 * Copyright 2013-2014 David Hadka
 *
 * This MEX function invokes the underlying C implementation of the
 * Borg MOEA.  This function should not be called directly; instead
 * use the borg function defined in the file borg.m.
 */
#ifdef OCTAVE
#include <cstdio>
#endif

#include <ctype.h>
#include <limits.h>
#include <time.h>
#include "mex.h"
#include "borg.h"

#ifndef OCTAVE
#include "matrix.h"
#endif

int numberOfVariables;
int numberOfObjectives;
int numberOfConstraints;
int maxEvaluations;
mxArray* epsilons = NULL;
mxArray* lowerBounds = NULL;
mxArray* upperBounds = NULL;
mxArray* functionHandle = NULL;
mxArray* parameters = NULL;
bool isTransposed = false;

/**
 * The callback function that invokes the MATLAB function handle.
 *
 * @param vars the array of decision variables
 * @param objs the array of objective values
 * @param constrs the array of constraints
 */
void callback(double* vars, double* objs, double* constrs) {
	mxArray *lhs[2];
	mxArray *rhs[2];

	rhs[0] = functionHandle;

	if (isTransposed) {
		rhs[1] = mxCreateDoubleMatrix(numberOfVariables, 1, mxREAL);
	} else {
		rhs[1] = mxCreateDoubleMatrix(1, numberOfVariables, mxREAL);
	}
    
	for (int i = 0; i < numberOfVariables; i++) {
		mxGetPr(rhs[1])[i] = vars[i];
	}

	if (numberOfConstraints > 0) {
		mexCallMATLAB(2, lhs, 2, rhs, "feval");
	} else {
		mexCallMATLAB(1, lhs, 2, rhs, "feval");
	}
    
	for (int i = 0; i < numberOfObjectives; i++) {
		objs[i] = mxGetPr(lhs[0])[i];
	}
    
	if (numberOfConstraints > 0) {
		for (int i = 0; i < numberOfConstraints; i++) {
			constrs[i] = mxGetPr(lhs[1])[i];
		}
        
		mxDestroyArray(lhs[1]);
	}

	mxDestroyArray(lhs[0]);
	mxDestroyArray(rhs[1]);
}

/**
 * Returns 1 if the two strings are equal ignoring case; 0 otherwise.
 *
 * @param array the character array from MATLAB
 * @param s2 the C string
 * @return 1 if the two strings are equal; 0 otherwise
 */
int matches(mxArray* array, const char* s2) {
	int i;
	int size = mxGetNumberOfElements(array);
	mxChar* s1 = mxGetChars(array);

	for (i=0; i<size; i++) {
		if ((s2[i] == '\0') ||
		    (tolower((unsigned)s1[i]) != tolower((unsigned)s2[i]))) {
			return false;
		}
	}

	return s2[size] == '\0';
}

/**
 * Finds and returns the value of the parameter with the given name; or
 * NULL if the parameter was not found.
 *
 * @param name the name of the parameter
 * @return the value of given parameter; or NULL if the parameter was
 *         not found
 */
mxArray* lookup(const char* name) {
	int i;
	int size = mxGetNumberOfElements(parameters);	

	for (i=0; i<size; i+=2) {
		mxArray* key = mxGetCell(parameters, i);

		if (!mxIsChar(key)) {
			mexErrMsgTxt("Parameter name is not a string.");
		}

		if (matches(key, name)) {
			return mxGetCell(parameters, i+1);
		}
	}

	return NULL;
}

/**
 * Finds and returns the value of the parameter with the given name; or
 * the defaultValue if the parameter was not found.
 *
 * @param name the name of the parameter
 * @param defaultValue the value to return if the parameter was not found
 * @return the value of the parameter; or defaultValue if the parameter
 *         was not found
 */
double lookupDouble(const char* name, double defaultValue) {
	mxArray* value = lookup(name);

	if (value == NULL) {
		return defaultValue;
	}
	
	if (!mxIsNumeric(value)) {
		mexErrMsgTxt("Parameter value is not numeric.");
	}
	
	if (mxGetNumberOfElements(value) != 1) {
		mexErrMsgTxt("Parameter value must be a scalar value.");
	}

	return mxGetScalar(value);
}

/**
 * Finds and returns the value of the parameter with the given name; or
 * the defaultValue if the parameter was not found.
 *
 * @param name the name of the parameter
 * @param defaultValue the value to return if the parameter was not found
 * @return the value of the parameter; or defaultValue if the parameter
 *         was not found
 */
int lookupInt(const char* name, int defaultValue) {
	return (int)lookupDouble(name, defaultValue);
}

/**
 * Finds and returns the value of the parameter with the given name; or
 * the defaultValue if the parameter was not found.
 *
 * @param name the name of the parameter
 * @param defaultValue the value to return if the parameter was not found
 * @return the value of the parameter; or defaultValue if the parameter
 *         was not found
 */
bool lookupBool(const char* name, bool defaultValue) {
	return (bool)lookupInt(name, defaultValue);
}

/**
 * The entry point for this MEX function being called from MATLAB.
 *
 * @param nlhs the number of outputs
 * @param plhs the outputs
 * @param nrhs the number of inputs
 * @param prhs the inputs
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	// validate arguments
	if (nrhs < 6) {
		mexErrMsgTxt("Requires six arguments (nvars, nobjs, nconstrs, objectiveFcn, nfe, epsilons).");
	}
    
	if (!mxIsNumeric(prhs[0])) {
		mexErrMsgTxt("Argument nvars must be an integer.");
	} else if (mxGetNumberOfElements(prhs[0]) != 1) {
		mexErrMsgTxt("Argument nvars must be a scalar value.");
	} else {
		numberOfVariables = (int)mxGetScalar(prhs[0]);
	}
    
	if (!mxIsNumeric(prhs[1])) {
		mexErrMsgTxt("Argument nobjs must be an integer.");
	} else if (mxGetNumberOfElements(prhs[1]) != 1) {
		mexErrMsgTxt("Argument nobjs must be a scalar value.");
	} else {
		numberOfObjectives = (int)mxGetScalar(prhs[1]);
	}
    
	if (!mxIsNumeric(prhs[2])) {
		mexErrMsgTxt("Argument nconstrs must be an integer.");
	} else if (mxGetNumberOfElements(prhs[2]) != 1) {
		mexErrMsgTxt("Argument nconstrs must be a scalar value.");
	} else {
		numberOfConstraints = (int)mxGetScalar(prhs[2]);
	}
    
	if(!mxIsClass(prhs[3], "function_handle")) {
		mexErrMsgTxt("Argument objectiveFcn must be a function handle.");
	} else {
		functionHandle = const_cast<mxArray*>(prhs[3]);
	}
    
	if (!mxIsNumeric(prhs[4])) {
		mexErrMsgTxt("Argument nfe must be an integer.");
	} else if (mxGetNumberOfElements(prhs[4]) != 1) {
		mexErrMsgTxt("Argument nfe must be a scalar value.");
	} else {
		maxEvaluations = (int)mxGetScalar(prhs[4]);
	}

	if (!mxIsNumeric(prhs[5])) {
		mexErrMsgTxt("Argument epsilons must be an array of numbers.");
	} else if (mxGetNumberOfElements(prhs[5]) != numberOfObjectives) {
		mexErrMsgTxt("Length of epsilons must match nobjs.");
	} else {
		epsilons = const_cast<mxArray*>(prhs[5]);
	}
    
	if (nrhs >= 7) {
		if (!mxIsNumeric(prhs[6])) {
			mexErrMsgTxt("Argument lowerBounds must be an array of numbers.");
		} else if (mxGetNumberOfElements(prhs[6]) != numberOfVariables) {
			mexErrMsgTxt("Length of lowerBounds must match nvars.");
		} else {
			lowerBounds = const_cast<mxArray*>(prhs[6]);
		}
	}
    
	if (nrhs >= 8) {
		if (!mxIsNumeric(prhs[7])) {
			mexErrMsgTxt("Argument upperBounds must be an array of numbers.");
		} else if (mxGetNumberOfElements(prhs[7]) != numberOfVariables) {
			mexErrMsgTxt("Length of upperBounds must match nvars.");
		} else {
			upperBounds = const_cast<mxArray*>(prhs[7]);
		}
	}

	if (nrhs >= 9) {
		if (!mxIsCell(prhs[8])) {
			mexErrMsgTxt("Argument parameters must be a cell array.");
		} else if (mxGetNumberOfElements(prhs[8]) % 2 != 0) {
			mexErrMsgTxt("Length of parameters must be even.");
		} else {
			parameters = const_cast<mxArray*>(prhs[8]);
		}
	}

	if (nrhs >= 10) {
		if (!mxIsNumeric(prhs[9]) && !mxIsLogical(prhs[9])) {
			mexErrMsgTxt("Argument isTransposed must be a logical value.");
		} else {
			isTransposed = (bool)mxGetScalar(prhs[9]);
		}
	}

	// start the timer
	clock_t start = clock();

	// setup the random number generator
	double rngstate = lookupDouble("rngstate", time(NULL));

	if ((rngstate > ULONG_MAX) || (rngstate < 0.0)) {
		mexErrMsgTxt("Argument rngstate is out of bounds.");
	}

	BORG_Random_seed((unsigned long)rngstate);

	// setup the problem
	BORG_Problem problem = BORG_Problem_create(numberOfVariables, numberOfObjectives, numberOfConstraints, callback);
    
	for (int i = 0; i < numberOfVariables; i++) {
		BORG_Problem_set_bounds(problem, i, lowerBounds ? mxGetPr(lowerBounds)[i] : 0.0, upperBounds ? mxGetPr(upperBounds)[i] : 1.0);
	}
    
	for (int i = 0; i < numberOfObjectives; i++) {
		BORG_Problem_set_epsilon(problem, i, mxGetPr(epsilons)[i]);
	}

	// setup the algorithm
	BORG_Operator pm = BORG_Operator_create("PM", 1, 1, 2, BORG_Operator_PM);
	BORG_Operator_set_parameter(pm, 0, lookupDouble("pm.rate", 1.0 / numberOfVariables));
	BORG_Operator_set_parameter(pm, 1, lookupDouble("pm.distributionIndex", 20.0));

	BORG_Operator sbx = BORG_Operator_create("SBX", 2, 2, 2, BORG_Operator_SBX);
	BORG_Operator_set_parameter(sbx, 0, lookupDouble("sbx.rate", 1.0));
	BORG_Operator_set_parameter(sbx, 1, lookupDouble("sbx.distributionIndex", 15.0));
	BORG_Operator_set_mutation(sbx, pm);

	BORG_Operator de = BORG_Operator_create("DE", 4, 1, 2, BORG_Operator_DE);
	BORG_Operator_set_parameter(de, 0, lookupDouble("de.crossoverRate", 0.1));
	BORG_Operator_set_parameter(de, 1, lookupDouble("de.stepSize", 0.5));
	BORG_Operator_set_mutation(de, pm);

	BORG_Operator um = BORG_Operator_create("UM", 1, 1, 1, BORG_Operator_UM);
	BORG_Operator_set_parameter(um, 0, lookupDouble("um.rate", 1.0 / numberOfVariables));

	BORG_Operator spx = BORG_Operator_create("SPX", lookupInt("spx.parents", 10), 
		lookupInt("spx.offspring", 2), 1, BORG_Operator_SPX);
	BORG_Operator_set_parameter(spx, 0, lookupDouble("spx.epsilon", 3.0));

	BORG_Operator pcx = BORG_Operator_create("PCX", lookupInt("pcx.parents", 10), 
		lookupInt("pcx.offspring", 2), 2, BORG_Operator_PCX);
	BORG_Operator_set_parameter(pcx, 0, lookupDouble("pcx.eta", 0.1));
	BORG_Operator_set_parameter(pcx, 1, lookupDouble("pcx.zeta", 0.1));

	BORG_Operator undx = BORG_Operator_create("UNDX", lookupInt("undx.parents", 10), 
		lookupInt("undx.offspring", 2), 2, BORG_Operator_UNDX);
	BORG_Operator_set_parameter(undx, 0, lookupDouble("undx.zeta", 0.5));
	BORG_Operator_set_parameter(undx, 1, lookupDouble("undx.eta", 0.35));

	BORG_Algorithm algorithm = BORG_Algorithm_create(problem, 6);
	BORG_Algorithm_set_operator(algorithm, 0, sbx);
	BORG_Algorithm_set_operator(algorithm, 1, de);
	BORG_Algorithm_set_operator(algorithm, 2, pcx);
	BORG_Algorithm_set_operator(algorithm, 3, spx);
	BORG_Algorithm_set_operator(algorithm, 4, undx);
	BORG_Algorithm_set_operator(algorithm, 5, um);

	BORG_Algorithm_set_initial_population_size(algorithm, lookupInt("initialPopulationSize", 100));
	BORG_Algorithm_set_minimum_population_size(algorithm, lookupInt("minimumPopulationSize", 100));
	BORG_Algorithm_set_maximum_population_size(algorithm, lookupInt("maximumPopulationSize", 10000));
	BORG_Algorithm_set_population_ratio(algorithm, 1.0 / lookupDouble("injectionRate", 0.25));
	BORG_Algorithm_set_selection_ratio(algorithm, lookupDouble("selectionRatio", 0.02));
	BORG_Algorithm_set_restart_mode(algorithm, static_cast<BORG_Restart>(lookupInt("restartMode", RESTART_DEFAULT)));
	BORG_Algorithm_set_max_mutation_index(algorithm, lookupInt("maxMutationIndex", 10));
	BORG_Algorithm_set_probability_mode(algorithm, static_cast<BORG_Probabilities>(lookupInt("probabilityMode", PROBABILITIES_DEFAULT)));

	// run the Borg MOEA
	int currentEvaluations = 0;
	int lastSnapshot = 0;
	int count = 0;
	int fields = 0;
	int frequency = lookupInt("frequency", 100);
	mxArray* runtimeData = NULL;
	mxArray* runtimeFields = NULL;

	if (nlhs >= 3) {
		if (lookupInt("restartMode", RESTART_DEFAULT) == RESTART_ADAPTIVE) {
			fields = 13;
		} else {
			fields = 12;
		}

		runtimeData = mxCreateDoubleMatrix(fields, maxEvaluations / frequency, mxREAL);
		runtimeFields = mxCreateCellMatrix(fields, 1);

		mxSetCell(runtimeFields, 0, mxCreateString("NFE"));
		mxSetCell(runtimeFields, 1, mxCreateString("ElapsedTime"));
		mxSetCell(runtimeFields, 2, mxCreateString("SBX"));
		mxSetCell(runtimeFields, 3, mxCreateString("DE"));
		mxSetCell(runtimeFields, 4, mxCreateString("PCX"));
		mxSetCell(runtimeFields, 5, mxCreateString("SPX"));
		mxSetCell(runtimeFields, 6, mxCreateString("UNDX"));
		mxSetCell(runtimeFields, 7, mxCreateString("UM"));
		mxSetCell(runtimeFields, 8, mxCreateString("Improvements"));
		mxSetCell(runtimeFields, 9, mxCreateString("Restarts"));
		mxSetCell(runtimeFields, 10, mxCreateString("PopulationSize"));
		mxSetCell(runtimeFields, 11, mxCreateString("ArchiveSize"));

		if (fields == 13) {
			mxSetCell(runtimeFields, 12, mxCreateString("MutationIndex"));
		}
	}

	while ((currentEvaluations = BORG_Algorithm_get_nfe(algorithm)) < maxEvaluations) {
		BORG_Algorithm_step(algorithm);

		if (runtimeData && (currentEvaluations - lastSnapshot >= frequency)) {
			mxGetPr(runtimeData)[fields*count + 0] = currentEvaluations;
			mxGetPr(runtimeData)[fields*count + 1] = (clock() - start) / (double)CLOCKS_PER_SEC;
			mxGetPr(runtimeData)[fields*count + 2] = BORG_Operator_get_probability(sbx);
			mxGetPr(runtimeData)[fields*count + 3] = BORG_Operator_get_probability(de);
			mxGetPr(runtimeData)[fields*count + 4] = BORG_Operator_get_probability(pcx);
			mxGetPr(runtimeData)[fields*count + 5] = BORG_Operator_get_probability(spx);
			mxGetPr(runtimeData)[fields*count + 6] = BORG_Operator_get_probability(undx);
			mxGetPr(runtimeData)[fields*count + 7] = BORG_Operator_get_probability(um);
			mxGetPr(runtimeData)[fields*count + 8] = BORG_Algorithm_get_number_improvements(algorithm);
			mxGetPr(runtimeData)[fields*count + 9] = BORG_Algorithm_get_number_restarts(algorithm);
			mxGetPr(runtimeData)[fields*count + 10] = BORG_Algorithm_get_population_size(algorithm);
			mxGetPr(runtimeData)[fields*count + 11] = BORG_Algorithm_get_archive_size(algorithm);

			if (fields == 13) {
				mxGetPr(runtimeData)[fields*count + 12] = BORG_Algorithm_get_mutation_index(algorithm);
			}

			count++;
			lastSnapshot = currentEvaluations;
		}
	}

	BORG_Archive result = BORG_Algorithm_get_result(algorithm);

	BORG_Operator_destroy(sbx);
	BORG_Operator_destroy(de);
	BORG_Operator_destroy(pm);
	BORG_Operator_destroy(um);
	BORG_Operator_destroy(spx);
	BORG_Operator_destroy(pcx);
	BORG_Operator_destroy(undx);
	BORG_Algorithm_destroy(algorithm);

	int size = BORG_Archive_get_size(result);

	// check if result contains feasible solutions
	if (size > 0) {
		if (BORG_Solution_violates_constraints(BORG_Archive_get(result, 0))) {
			size = 0;
		}
	}
    
	// generate results to send back to MATLAB
	if (nlhs >= 1) {
		plhs[0] = mxCreateDoubleMatrix(size, numberOfVariables, mxREAL);

		for (int i = 0; i < size; i++) {
			BORG_Solution solution = BORG_Archive_get(result, i);
            
			for (int j = 0; j < numberOfVariables; j++) {
				mxGetPr(plhs[0])[size*j + i] = BORG_Solution_get_variable(solution, j);
			}
		}
	}
    
	if (nlhs >= 2) {
		plhs[1] = mxCreateDoubleMatrix(size, numberOfObjectives, mxREAL);
        
		for (int i = 0; i < size; i++) {
			BORG_Solution solution = BORG_Archive_get(result, i);
            
			for (int j = 0; j < numberOfObjectives; j++) {
				mxGetPr(plhs[1])[size*j + i] = BORG_Solution_get_objective(solution, j);
			}
		}
	}

	if (nlhs >= 3) {
		mxSetN(runtimeData, count-1);

		const char* fieldNames[] = {"fields", "values"};
		plhs[2] = mxCreateStructMatrix(1, 1, 2, fieldNames);
		mxSetField(plhs[2], 0, fieldNames[0], runtimeFields);
		mxSetField(plhs[2], 0, fieldNames[1], runtimeData);
	}
    
	// free resources
	BORG_Archive_destroy(result);
	BORG_Problem_destroy(problem);
}
