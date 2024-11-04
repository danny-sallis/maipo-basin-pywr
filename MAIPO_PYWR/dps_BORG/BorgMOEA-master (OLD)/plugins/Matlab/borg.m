function [vars, objs, runtime] = borg(nvars, nobjs, nconstrs, objectiveFcn, NFE, epsilons, lowerBounds, upperBounds, parameters)
%BORG Optimize a multiobjective problem using the Borg MOEA.
%  BORG(nvars, nobjs, nconstrs, objectiveFcn, NFE, epsilons)
%  BORG(nvars, nobjs, nconstrs, objectiveFcn, NFE, epsilons, _
%       lowerBounds, upperBounds, _
%  BORG(nvars, nobjs, nconstrs, objectiveFcn, NFE, epsilons, _
%       lowerBounds, upperBounds, parameters)
%  [vars, objs] = BORG(...)
%  [vars, objs, runtime] = BORG(...)
%
%  The multiobjective problem is defined by objectiveFcn, which is the handle
%  of a function of the form:
%      [objs] = objectiveFcn(vars)
%  where vars is the array of decision variables and objs is an array of one or
%  more objective values.  If N decision variables are required by the problem,
%  then vars will either be a 1xN or Nx1 array.  If lowerBounds is 1xN, then
%  vars will be 1xN.  Alternatively, if lowerBounds is Nx1, then vars will be
%  Nx1.
%
%  If the problem is constrained, the objective function should follow the form:
%      [objs, constrs] = objectiveFcn(vars)
%  where constrs is an array of one or more constraints.  Constraint violations
%  are identified by non-zero values.  Therefore, constrs must contain all zeros
%  if the solution is feasible.
%
%  NFE, the number of objective function evaluations, defines how many times
%  the Borg MOEA can invoke objectiveFcn.  Once the NFE limit is reached, the
%  algorithm terminates and returns the result.
%
%  Epsilons control the resolution of solutions discovered by the Borg MOEA.
%  Smaller epsilon values result in fine-grained Pareto sets, and larger
%  epsilons in coarse-grained sets.  If the multiobjective problem defines M
%  objectives, then epsilons should be a 1xM array of epsilon values.
%
%  All of the parameters defined by the C Borg MOEA can be overridden with the
%  optional parameters arguments.  Parameters are defined in a cell array whose
%  elements alternate between the parameter name and parameter value.  For
%  example:
%      parameters = {'SBX.rate', 0.9, 'SBX.distributionIndex', 15.0};
%
%  BORG returns the decision variables and objectives of the Pareto optimal
%  solutions.  Each row in vars and objs corresponds to a feasible solution.If
%  no feasible Pareto optimal solutions were discovered, then empty matrices are
%  returned.  
%
%  If the runtime output is requested, runtime dynamics will be collected
%  throughout the run.  The frequency at which the runtime dynamics are
%  collected is controlled by the 'frequency' parameter.
%
%  Example:
%      DTLZ2 is a 11 decision variable, 2 objective problem.  The decision
%      variables range between [0, 1].  The following command solves the 
%      DTLZ2 problem using 100,000 objective function evaluations.
%          [vars, objs] = borg(11, 2, 0, @DTLZ2, 100000, 0.01*ones(1,2), _
%                              zeros(1,11), ones(1,11));
%
% Please cite the following paper in any works that use or are derived
% from this program.
%
%      Hadka, D. and Reed, P. (2013).  "Borg: An Auto-Adaptive Many-Objective
%      Evolutionary Computing Framework."  Evolutionary Computation,
%      21(2):231-259.
%
%  Copyright 2013-2014 David Hadka

if ~exist('nativeborg')
	error('Unable to find the nativeborg MEX-function, please follow the setup instructions to compile the MEX-function');
end

if nargin < 6
	error('Requires at least six arguments');
end

if nvars < 1
	error('Requires at least one decision variable');
end

if nobjs < 1
	error('Requires at least one objective');
end

if nconstrs < 0
	error('Number of constraints can not be negative');
end

if isa(objectiveFcn, 'function_handle')
	%already a function handle, ok
elseif isa(objectiveFcn, 'char')
	objectiveFcn = str2func(objectiveFcn);
else
	error('Objective function is not a valid function name or function handle');
end

if NFE < 1
	error('Requires a positive number of objective function evaluations (NFE)');
end

if numel(epsilons) ~= nobjs
	error('Length of epsilon array must match the number of objectives');
end

if nargin < 8
    if (nargin == 7)
        error('Must specify both lower and upper bounds')
    end
    
	lowerBounds = zeros(1, nvars);
	upperBounds = ones(1, nvars);
else
	if numel(lowerBounds) ~= nvars
		error('Length of lower bounds must match the number of decision variables')
	end

	if numel(upperBounds) ~= nvars
		error('Length of upper bounds must match the number of decision variables')
	end

	if size(lowerBounds, 1) ~= 1 && size(lowerBounds, 2) ~= 1
		error('Lower and upper bound arrays must not be multidimensional');
	end
end

transposed = size(lowerBounds, 1) > 1;

if nargin < 9
	parameters = {};
end

if nargout >= 3
	[vars, objs, runtime] = nativeborg(nvars, nobjs, nconstrs, objectiveFcn, NFE, epsilons, lowerBounds, upperBounds, parameters, transposed);
elseif nargout >= 2
	[vars, objs] = nativeborg(nvars, nobjs, nconstrs, objectiveFcn, NFE, epsilons, lowerBounds, upperBounds, parameters, transposed);
else
	[vars] = nativeborg(nvars, nobjs, nconstrs, objectiveFcn, NFE, epsilons, lowerBounds, upperBounds, parameters, transposed);
end
