using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;

/// <summary>C# Interface for the Borg MOEA.</summary>
/// <remarks>
/// This C# interface for the Borg MOEA allows C# functions to be optimized by the native Borg MOEA
/// library.  To use this interface, the Borg MOEA library, typically called borg.dll, must be located
/// in the same folder as the compiled C# code or listed on your system path.
///
/// Please cite the following paper in any works that use or are derived from this program.
///
///     Hadka, D. and Reed, P. (2013).  "Borg: An Auto-Adaptive Many-Objective Evolutionary Computing
///     Framework."  Evolutionary Computation, 21(2):231-259.
///
/// Copyright 2014 David Hadka
/// </remarks>
namespace BorgMOEA
{

    /// <summary>Indicates if a point is dominated, non-dominated, or dominates other points.</summary>
    public enum BORG_Dominance
    {
        /// <summary>The point epsilon-box dominates one or more other points.</summary>
        DOMINATES = -2,

        /// <summary>The point epsilon-box dominates another point within the same epsilon-box.</summary>
        DOMINATES_SAME_BOX = -1,

        /// <summary>The point neither dominates nor is dominated by any other point.</summary>
        NONDOMINATED = 0,

        /// <summary>The point is epsilon-box dominated by another point within the same epsilon-box.</summary>
        DOMINATED_SAME_BOX = 1,

        /// <summary>The point is epsilon-box dominated by one or more other points.</summary>
        DOMINATED = 2,
    }

    /// <summary>Controls the mutation rate during restarts.</summary>
    public enum BORG_Restart
    {
        /// <summary>The mutation rate is fixed at 1/numberOfVariables.</summary>
        RESTART_DEFAULT = 0,

        /// <summary>The mutation rate is fixed at 100%.</summary>
        RESTART_RANDOM = 1,

        /// <summary>The mutation rates are uniformly sampled between 1/numberOfVariables to 100%.</summary>
        RESTART_RAMPED = 2,

        /// <summary>The mutation rate adapts based on success of previous restarts.</summary>
        RESTART_ADAPTIVE = 3,

        /// <summary>Similar to ADAPTIVE, except the rate is inverted.</summary>
        RESTART_INVERTED = 4,
    }

    /// <summary>Controls how operator probabilities are adapted.</summary>
    public enum BORG_Probabilities
    {
        /// <summary>Operator probabilities based on archive membership.</summary>
        PROBABILITIES_DEFAULT = 0,

        /// <summary>Operator probabilities based on recency (tracks recent additions to archive).</summary>
        PROBABILITIES_RECENCY = 1,

        /// <summary>Operator probabilities based on archive membership and recency.</summary>
        PROBABILITIES_BOTH = 2,

        /// <summary>Favors archive membership, but uses recency if insufficient archive size.</summary>
        PROBABILITIES_ADAPTIVE = 3,
    }

    /// <summary>
    /// Class representing a solution to an optimization problem, storing the decision variables,
    /// objectives, and constraint values.
    /// </summary>
    public class Solution
    {
        /// <summary>Pointer to the underlying C object.</summary>
        private IntPtr solution;

        /// <summary>Creates a new Solution given a pointer to an underlying C object.</summary>
        /// <param name="solution">Pointer to the underyling C object.</param>
        public Solution(IntPtr solution)
        {
            this.solution = solution;
        }

        /// <summary>Returns the value of a decision variable.</summary>
        /// <param name="index">The index of the decision variable to return.</param>
        /// <returns>The value of the decision variable.</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown when <code>index < 0 || index >= GetNumberOfVariables()</code>.</exception>
        public double GetVariable(int index)
        {
            if ((index < 0) || (index > GetNumberOfVariables()))
            {
                throw new IndexOutOfRangeException();
            }
            else
            {
                return BorgLibrary.BORG_Solution_get_variable(solution, index);
            }
        }

        /// <summary>Returns the value of an objective.</summary>
        /// <param name="index">The index of the objective to return.</param>
        /// <returns>The value of the objective.</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown when <code>index < 0 || index >= GetNumberOfObjectives()</code>.</exception>
        public double GetObjective(int index)
        {
            if ((index < 0) || (index > GetNumberOfObjectives()))
            {
                throw new IndexOutOfRangeException();
            }
            else
            {
                return BorgLibrary.BORG_Solution_get_objective(solution, index);
            }
        }

        /// <summary>Returns the value of a constraint.</summary>
        /// <param name="index">The index of the constraint to return.</param>
        /// <returns>The value of the constraint.</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown when <code>index < 0 || index >= GetNumberOfConstraints()</code>.</exception>
        public double GetConstraint(int index)
        {
            if ((index < 0) || (index > GetNumberOfConstraints()))
            {
                throw new IndexOutOfRangeException();
            }
            else
            {
                return BorgLibrary.BORG_Solution_get_constraint(solution, index);
            }
        }

        /// <summary>Returns the number of decision variables.</summary>
        /// <returns>The number of decision variable.</returns>
        public int GetNumberOfVariables()
        {
            IntPtr problem = BorgLibrary.BORG_Solution_get_problem(solution);
            return BorgLibrary.BORG_Problem_number_of_variables(problem);
        }

        /// <summary>Returns the number of objectives.</summary>
        /// <returns>The number of objectives.</returns>
        public int GetNumberOfObjectives()
        {
            IntPtr problem = BorgLibrary.BORG_Solution_get_problem(solution);
            return BorgLibrary.BORG_Problem_number_of_objectives(problem);
        }

        /// <summary>Returns the number of constraints.</summary>
        /// <returns>The number of constraints.</returns>
        public int GetNumberOfConstraints()
        {
            IntPtr problem = BorgLibrary.BORG_Solution_get_problem(solution);
            return BorgLibrary.BORG_Problem_number_of_constraints(problem);
        }

        /// <summary>Returns true if this solution violates any constraint; false otherwise.</summary>
        /// <returns>true if this solution violates any constraint; false otherwise.</returns>
        public bool ViolatesConstraints()
        {
            return BorgLibrary.BORG_Solution_violates_constraints(solution) != 0;
        }
    }

    /// <summary>Class representing the results of optimization (a Pareto optimal set).</summary>
    public class Result : IEnumerable<Solution>
    {
        /// <summary>Pointer to the underlying C object.</summary>
        private IntPtr archive;

        /// <summary>Optional field recording runtime statistics.</summary>
        private List<Dictionary<string, double>> statistics;

        /// <summary>Creates a new Result given a pointer to an underlying C object.</summary>
        /// <param name="archive">Pointer to the underyling C object.</param>
        public Result(IntPtr archive)
            : this(archive, null)
        {
            // do nothing
        }

        /// <summary>Creates a new Result given a pointer to an underlying C object.</summary>
        /// <param name="archive">Pointer to the underyling C object.</param>
        /// <param name="statistics">Optional field recording runtime statistics.</param>
        public Result(IntPtr archive, List<Dictionary<string, double>> statistics)
        {
            this.archive = archive;
            this.statistics = statistics;
        }

        /// <summary>Deletes this Result object</summary>
        ~Result()
        {
            BorgLibrary.BORG_Archive_destroy(archive);
        }

        /// <summary>Returns true if this result contains runtime statistics; false otherwise.</summary>
        /// <returns>true if this result contains runtime statistics; false otherwise.</returns>
        public bool HasStatistics()
        {
            return statistics != null;
        }

        /// <summary>Returns the runtime statistics.</summary>
        /// <returns>The runtime statistics.</summary>
        public List<Dictionary<string, double>> GetStatistics()
        {
            return statistics;
        }

        /// <summary>Returns the number of solutions contained in this result.</summary>
        /// <returns>The number of solutions contained in this result.</returns>
        public int Size()
        {
            return BorgLibrary.BORG_Archive_get_size(archive);
        }

        /// <summary>Returns a solution contained in this result.</summary>
        /// <param name="index">The index of the solution to return.</param>
        /// <returns>The selected solution.</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown when <code>index < 0 || index >= Size()</code>.</exception>
        public Solution Get(int index)
        {
            if ((index < 0) || (index >= Size()))
            {
                throw new IndexOutOfRangeException();
            }
            else
            {
                return new Solution(BorgLibrary.BORG_Archive_get(archive, index));
            }
        }

        /// <summary>Creates an iterator over the solutions in this result.</summary>
        /// <returns>The iterator over the solutions in this result.</returns>
        public IEnumerator<Solution> GetEnumerator()
        {
            return new ResultEnumerator(this);
        }

        /// <summary>Creates an iterator over the solutions in this result.</summary>
        /// <returns>The iterator over the solutions in this result.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    /// <summary>Iterates over the solutions in a result object.</summary>
    public class ResultEnumerator : IEnumerator<Solution>
    {
        /// <summary>The result object being enumerated.</summary>
        private Result result;

        /// <summary>The current index.</summary>
        private int index;

        /// <summary>Creates a new iterator over the solutions in the given result object.</summary>
        /// <param name="result">The result object to enumerate.</summary>
        public ResultEnumerator(Result result)
        {
            this.result = result;
            Reset();
        }

        public void Dispose()
        {
            // do nothing
        }

        public bool MoveNext()
        {
            index++;
            return index < result.Size();
        }

        public void Reset()
        {
            index = -1;
        }

        public Solution Current
        {
            get
            {
                try
                {
                    return result.Get(index);
                }
                catch (IndexOutOfRangeException)
                {
                    throw new InvalidOperationException();
                }
            }
        }

        object IEnumerator.Current
        {
            get
            {
                return Current;
            }
        }
    }

    /// <summary>Class representing the Borg Multiobjective Evolutionary Algorithm (Borg MOEA).</summary>
    public class Borg
    {
        /// <summary>Pointer to the underlying C object representing the optimization problem.</summary>
        private IntPtr problem;

        /// <summary>C# wrapper around the optimization problem.</summary>
        private FunctionWrapper wrapper;

        /// <summary>The delegate used to invoke the native function from C#.</summary>
        private NativeObjectiveFunction nativeFunction;

        /// <summary>Garbage collection handle for function delegate.</summary>
        private GCHandle nativeFunctionHandle;

        /// <summary>Pointer to function in unmanaged memory that can be safely passed to C.</summary>
        private IntPtr nativeFunctionPointer;

        /// <summary>Dictionary of parameters for the optimization algorithm.</summary>
        private Dictionary<string, double> settings;

        /// <summary>Initiaize the random number generator.</summary>
        static Borg()
        {
            BorgLibrary.BORG_Random_seed((uint)(DateTime.Now - DateTime.MinValue).TotalMilliseconds);
        }

        /// <summary>Constructs a new instance of the Borg MOEA for optimizing the given function.</summary>
        /// <param name="numberOfVariables">The number of decision variables.</param>
        /// <param name="numberOfObjectives">The number of objectives.</param>
        /// <param name="numberOfConstraints">The number of constraints.</param>
        /// <param name="function">The multiobjective function being optimized (minimized).</param>
        /// <exception cref="ArgumentException">Thrown if any input is invalid.</exception>
        public Borg(int numberOfVariables, int numberOfObjectives, int numberOfConstraints, ObjectiveFunction function)
        {
            if (numberOfVariables < 1)
            {
                throw new ArgumentException("Requires at least one decision variable");
            }

            if (numberOfObjectives < 1)
            {
                throw new ArgumentException("Requires at least one objective");
            }

            if (numberOfConstraints < 0)
            {
                throw new ArgumentException("Number of constraints can not be negative");
            }

            if (function == null)
            {
                throw new ArgumentException("Function is null");
            }

            wrapper = new FunctionWrapper(numberOfVariables, numberOfObjectives, numberOfConstraints, function);
            nativeFunction = new NativeObjectiveFunction(wrapper.Evaluate);
            nativeFunctionHandle = GCHandle.Alloc(nativeFunction);
            nativeFunctionPointer = Marshal.GetFunctionPointerForDelegate(nativeFunction);
            problem = BorgLibrary.BORG_Problem_create(numberOfVariables, numberOfObjectives, numberOfConstraints, nativeFunctionPointer);
            settings = new Dictionary<string, double>();
        }

        /// <summary>Destroy this instance of the Borg MOEA.</summary>
        ~Borg()
        {
            if (!problem.Equals(null))
            {
                BorgLibrary.BORG_Problem_destroy(problem);
                nativeFunctionHandle.Free();
            }
        }

        /// <summary>Returns the number of decision varibles.</summary>
        /// <returns>The number of decision variables.</returns>
        public int GetNumberOfVariables()
        {
            return BorgLibrary.BORG_Problem_number_of_variables(problem);
        }

        /// <summary>Returns the number of objectives.</summary>
        /// <returns>The number of objectives.</returns>
        public int GetNumberOfObjectives()
        {
            return BorgLibrary.BORG_Problem_number_of_objectives(problem);
        }

        /// <summary>Returns the number of constraints.</summary>
        /// <returns>The number of constraints.</returns>
        public int GetNumberOfConstraints()
        {
            return BorgLibrary.BORG_Problem_number_of_constraints(problem);
        }

        /// <summary>Sets the lower and upper bounds for all decision variables.</summary>
        /// <param name="lowerBounds">The lower bounds of the decision variables.</param>
        /// <param name="upperBounds">The upper bounds of the decision variables.</param>
        /// <exception cref="ArgumentException">Thrown if the length of the arrays do not match the number of variables.</exception>
        public void SetBounds(double[] lowerBounds, double[] upperBounds)
        {
            if (lowerBounds.Length != upperBounds.Length)
            {
                throw new ArgumentException("Incorrect number of bounds specified");
            }

            if (lowerBounds.Length != GetNumberOfVariables())
            {
                throw new ArgumentException("Incorrect number of bounds specified");
            }

            for (int i = 0; i < lowerBounds.Length; i++)
            {
                SetBounds(i, lowerBounds[i], upperBounds[i]);
            }
        }

        /// <summary>Sets the lower and upper bounds for a decision variable.</summary>
        /// <param name="index">The index of the decision variable.</param>
        /// <param name="lowerBound">The lower bound of the decision variable.</param>
        /// <param name="upperBound">The upper bound of the decision variable.</param>
        /// <exception cref="IndexOutOfRangeException">Thrown if <code>index < 0 || index >= GetNumberOfVariables()</code></exception>
        public void SetBounds(int index, double lowerBound, double upperBound)
        {
            if ((index < 0) || (index >= GetNumberOfVariables()))
            {
                throw new IndexOutOfRangeException();
            }
            else
            {
                BorgLibrary.BORG_Problem_set_bounds(problem, index, lowerBound, upperBound);
            }
        }

        /// <summary>Sets all epsilon values for the objectives.</summary>
        /// <param name="epsilons">The epsilon values for the objectives.</param>
        /// <exception cref="ArgumentException">Thrown if the length of the array does not match the number of objectives.</exception>
        public void SetEpsilons(double[] epsilons)
        {
            if (epsilons.Length != GetNumberOfObjectives())
            {
                throw new ArgumentException("Incorrect number of epsilons specified");
            }

            for (int i = 0; i < epsilons.Length; i++)
            {
                SetEpsilon(i, epsilons[i]);
            }
        }

        /// <summary>Sets the epsilon values for an objective.</summary>
        /// <param name="index">The index of the objective.</param>
        /// <param name="epsilon">The epsilon value for the objective.</param>
        /// <exception cref="IndexOutOfRangeException">Thrown if <code>index < 0 || index >= GetNumberOfObjectives()</code></exception>
        public void SetEpsilon(int index, double epsilon)
        {
            if ((index < 0) || (index >= GetNumberOfObjectives()))
            {
                throw new IndexOutOfRangeException();
            }
            else
            {
                BorgLibrary.BORG_Problem_set_epsilon(problem, index, epsilon);
            }
        }

        private double LookupDouble(string name, double defaultValue)
        {
            if (settings.ContainsKey(name))
            {
                return settings[name];
            }
            else
            {
                return defaultValue;
            }
        }

        private int LookupInt(string name, int defaultValue)
        {
            if (settings.ContainsKey(name))
            {
                return (int)settings[name];
            }
            else
            {
                return defaultValue;
            }
        }

        /// <summary>Overrides the default value for a parameter.</summary>
        /// <remarks>
        /// Valid parameters include:
        /// <list type="bullet">
        /// <item><description>pm.rate - The mutation rate for polynomial mutation</description></item>
        /// <item><description>pm.distributionIndex - The distribution index for polynomial mutation</description></item>
        /// <item><description>sbx.rate - The crossover rate for simulated binary crossover</description></item>
        /// <item><description>sbx.distributionIndex - The distribution index for simulated binary crossover</description></item>
        /// <item><description>de.crossoverRate - The crossover rate for differential evolution</description></item>
        /// <item><description>de.stepSize - The step size for differential evolution</description></item>
        /// <item><description>um.rate - The mutation rate for uniform mutation</description></item>
        /// <item><description>spx.parents - The number of parents used by simplex crossover</description></item>
        /// <item><description>spx.offspring - The number of offspring generated by simplex crossover</description></item>
        /// <item><description>spx.epsilon - The expansion factor used by simplex crossover</description></item>
        /// <item><description>pcx.parents - The number of parents used by parent-centric crossover</description></item>
        /// <item><description>pcx.offspring - The number of offspring generated by parent-centric crossover</description></item>
        /// <item><description>pcx.eta - The standard deviation of the normal distribution controlling the spread of solutions in the direction of the selected parent</description></item>
        /// <item><description>pcx.zeta - The standard deviation of the normal distribution controlling the spread of solutions in the directions defined by the remaining parents</description></item>
        /// <item><description>undx.parents - The number of parents used by unimodal normal distribution crossover</description></item>
        /// <item><description>undx.offspring - The number of offspring generated by unimal normal distribution crossover</description></item>
        /// <item><description>undx.zeta - The standard deviation of the normal distribution controlling the spread of solutions in the orthogonal directions defined by the parents</description></item>
        /// <item><description>undx.eta - The standard deviation of the normal distribution controlling the spread of solutions in the remaining orthogonal directions not defined by the parents</description></item>
        /// <item><description>initialPopulationSize - The initial population size</description></item>
        /// <item><description>minimumPopulationSize - The minimum population size</description></item>
        /// <item><description>maximumPopulationSize - The maximum population size</description></item>
        /// <item><description>injectionRate - The injection rate during restarts</description></item>
        /// <item><description>selectionRatio - The selection ratio (size of tournament / population size)</description></item>
        /// <item><description>restartMode - Controls the mutation rate during restarts</description></item>
        /// <item><description>maxMutationIndex - Controls how quickly the mutation rate during restarts increases (only used by RESTART_ADAPTIVE)</description></item>
        /// <item><description>probabilityMode - Controls how Borg MOEA determines the probability of applying each operator</description></item>
        /// <item><description>frequency - The frequency (in NFE) to record runtime statistics</description></item>
        /// </list>
        /// </remarks>
        /// <param name="name">The name of the parameter.</param>
        /// <param name="value">The value of the parameter.</param>
        public void SetParameter(string name, double value)
        {
            settings[name] = value;
        }

        /// <summary>Overrides the default value for a parameter.  See the other <code>SetParameter</code> method for valid parameters.</summary>
        /// <param name="name">The name of the parameter.</param>
        /// <param name="value">The value of the parameter.</param>
        public void SetParameter(string name, int value)
        {
            settings[name] = value;
        }

        /// <summary>Optimizes the multiobjective function.</summary>
        /// <param name="maxEvaluations">The maximum number of evaluations to run the Borg MOEA.</param>
        /// <returns>The optimizaton result (the Pareto optimal set).</returns>
        public Result Solve(int maxEvaluations)
        {
            Stopwatch timer = new Stopwatch();
            timer.Start();

            OperatorFunction pmfcn = new OperatorFunction(BorgLibrary.BORG_Operator_PM);
            GCHandle pmhdl = GCHandle.Alloc(pmfcn);
            IntPtr pmptr = Marshal.GetFunctionPointerForDelegate(pmfcn);

            IntPtr pm = BorgLibrary.BORG_Operator_create("PM", 1, 1, 2, pmptr);
            BorgLibrary.BORG_Operator_set_parameter(pm, 0, LookupDouble("pm.rate", 1.0 / GetNumberOfVariables()));
            BorgLibrary.BORG_Operator_set_parameter(pm, 1, LookupDouble("pm.distributionIndex", 20.0));

            OperatorFunction sbxfcn = new OperatorFunction(BorgLibrary.BORG_Operator_SBX);
            GCHandle sbxhdl = GCHandle.Alloc(sbxfcn);
            IntPtr sbxptr = Marshal.GetFunctionPointerForDelegate(sbxfcn);

            IntPtr sbx = BorgLibrary.BORG_Operator_create("SBX", 2, 2, 2, sbxptr);
            BorgLibrary.BORG_Operator_set_parameter(sbx, 0, LookupDouble("sbx.rate", 1.0));
            BorgLibrary.BORG_Operator_set_parameter(sbx, 1, LookupDouble("sbx.distributionIndex", 15.0));
            BorgLibrary.BORG_Operator_set_mutation(sbx, pm);

            OperatorFunction defcn = new OperatorFunction(BorgLibrary.BORG_Operator_DE);
            GCHandle dehdl = GCHandle.Alloc(defcn);
            IntPtr deptr = Marshal.GetFunctionPointerForDelegate(defcn);

            IntPtr de = BorgLibrary.BORG_Operator_create("DE", 4, 1, 2, deptr);
            BorgLibrary.BORG_Operator_set_parameter(de, 0, LookupDouble("de.crossoverRate", 0.1));
            BorgLibrary.BORG_Operator_set_parameter(de, 1, LookupDouble("de.stepSize", 0.5));
            BorgLibrary.BORG_Operator_set_mutation(de, pm);

            OperatorFunction umfcn = new OperatorFunction(BorgLibrary.BORG_Operator_UM);
            GCHandle umhdl = GCHandle.Alloc(umfcn);
            IntPtr umptr = Marshal.GetFunctionPointerForDelegate(umfcn);

            IntPtr um = BorgLibrary.BORG_Operator_create("UM", 1, 1, 1, umptr);
            BorgLibrary.BORG_Operator_set_parameter(um, 0, LookupDouble("um.rate", 1.0 / GetNumberOfVariables()));

            OperatorFunction spxfcn = new OperatorFunction(BorgLibrary.BORG_Operator_SPX);
            GCHandle spxhdl = GCHandle.Alloc(spxfcn);
            IntPtr spxptr = Marshal.GetFunctionPointerForDelegate(spxfcn);

            IntPtr spx = BorgLibrary.BORG_Operator_create("SPX", LookupInt("spx.parents", 10), LookupInt("spx.offspring", 2), 1, spxptr);
            BorgLibrary.BORG_Operator_set_parameter(spx, 0, LookupDouble("spx.epsilon", 3.0));

            OperatorFunction pcxfcn = new OperatorFunction(BorgLibrary.BORG_Operator_PCX);
            GCHandle pcxhdl = GCHandle.Alloc(pcxfcn);
            IntPtr pcxptr = Marshal.GetFunctionPointerForDelegate(pcxfcn);

            IntPtr pcx = BorgLibrary.BORG_Operator_create("PCX", LookupInt("pcx.parents", 10), LookupInt("pcx.offspring", 2), 2, pcxptr);
            BorgLibrary.BORG_Operator_set_parameter(pcx, 0, LookupDouble("pcx.eta", 0.1));
            BorgLibrary.BORG_Operator_set_parameter(pcx, 1, LookupDouble("pcx.zeta", 0.1));

            OperatorFunction undxfcn = new OperatorFunction(BorgLibrary.BORG_Operator_UNDX);
            GCHandle undxhdl = GCHandle.Alloc(undxfcn);
            IntPtr undxptr = Marshal.GetFunctionPointerForDelegate(undxfcn);

            IntPtr undx = BorgLibrary.BORG_Operator_create("UNDX", LookupInt("undx.parents", 10), LookupInt("undx.offspring", 2), 2, undxptr);
            BorgLibrary.BORG_Operator_set_parameter(undx, 0, LookupDouble("undx.zeta", 0.5));
            BorgLibrary.BORG_Operator_set_parameter(undx, 1, LookupDouble("undx.eta", 0.35));

            IntPtr algorithm = BorgLibrary.BORG_Algorithm_create(problem, 6);
            BorgLibrary.BORG_Algorithm_set_operator(algorithm, 0, sbx);
            BorgLibrary.BORG_Algorithm_set_operator(algorithm, 1, de);
            BorgLibrary.BORG_Algorithm_set_operator(algorithm, 2, pcx);
            BorgLibrary.BORG_Algorithm_set_operator(algorithm, 3, spx);
            BorgLibrary.BORG_Algorithm_set_operator(algorithm, 4, undx);
            BorgLibrary.BORG_Algorithm_set_operator(algorithm, 5, um);

            BorgLibrary.BORG_Algorithm_set_initial_population_size(algorithm, LookupInt("initialPopulationSize", 100));
            BorgLibrary.BORG_Algorithm_set_minimum_population_size(algorithm, LookupInt("minimumPopulationSize", 100));
            BorgLibrary.BORG_Algorithm_set_maximum_population_size(algorithm, LookupInt("maximumPopulationSize", 10000));
            BorgLibrary.BORG_Algorithm_set_population_ratio(algorithm, 1.0 / LookupDouble("injectionRate", 0.25));
            BorgLibrary.BORG_Algorithm_set_selection_ratio(algorithm, LookupDouble("selectionRatio", 0.02));
            BorgLibrary.BORG_Algorithm_set_restart_mode(algorithm, (BORG_Restart)LookupInt("restartMode", (int)BORG_Restart.RESTART_DEFAULT));
            BorgLibrary.BORG_Algorithm_set_max_mutation_index(algorithm, LookupInt("maxMutationIndex", 10));
            BorgLibrary.BORG_Algorithm_set_probability_mode(algorithm, (BORG_Probabilities)LookupInt("probabilityMode", (int)BORG_Probabilities.PROBABILITIES_DEFAULT));

            List<Dictionary<string, double>> statistics = null;
            int lastSnapshot = 0;
            int frequency = LookupInt("frequency", 0);

            if (frequency > 0)
            {
                statistics = new List<Dictionary<string, double>>();
            }

            while (BorgLibrary.BORG_Algorithm_get_nfe(algorithm) < maxEvaluations)
            {
                BorgLibrary.BORG_Algorithm_step(algorithm);

                int currentEvaluations = BorgLibrary.BORG_Algorithm_get_nfe(algorithm);

                if ((statistics != null) && (currentEvaluations - lastSnapshot >= frequency))
                {
                    Dictionary<string, double> entry = new Dictionary<string, double>();
                    entry["NFE"] = currentEvaluations;
                    entry["ElapsedTime"] = timer.ElapsedMilliseconds;
                    entry["SBX"] = BorgLibrary.BORG_Operator_get_probability(sbx);
                    entry["DE"] = BorgLibrary.BORG_Operator_get_probability(de);
                    entry["PCX"] = BorgLibrary.BORG_Operator_get_probability(pcx);
                    entry["SPX"] = BorgLibrary.BORG_Operator_get_probability(spx);
                    entry["UNDX"] = BorgLibrary.BORG_Operator_get_probability(undx);
                    entry["UM"] = BorgLibrary.BORG_Operator_get_probability(um);
                    entry["Improvements"] = BorgLibrary.BORG_Algorithm_get_number_improvements(algorithm);
                    entry["Restarts"] = BorgLibrary.BORG_Algorithm_get_number_restarts(algorithm);
                    entry["PopulationSize"] = BorgLibrary.BORG_Algorithm_get_population_size(algorithm);
                    entry["Improvements"] = BorgLibrary.BORG_Algorithm_get_archive_size(algorithm);

                    if ((BORG_Restart)LookupInt("restartMode", (int)BORG_Restart.RESTART_DEFAULT) == BORG_Restart.RESTART_ADAPTIVE)
                    {
                        entry["MutationIndex"] = BorgLibrary.BORG_Algorithm_get_mutation_index(algorithm);
                    }

                    statistics.Add(entry);
                    lastSnapshot = currentEvaluations;
                }
            }

            IntPtr result = BorgLibrary.BORG_Algorithm_get_result(algorithm);

            BorgLibrary.BORG_Operator_destroy(sbx);
            BorgLibrary.BORG_Operator_destroy(de);
            BorgLibrary.BORG_Operator_destroy(pm);
            BorgLibrary.BORG_Operator_destroy(um);
            BorgLibrary.BORG_Operator_destroy(spx);
            BorgLibrary.BORG_Operator_destroy(pcx);
            BorgLibrary.BORG_Operator_destroy(undx);
            BorgLibrary.BORG_Algorithm_destroy(algorithm);

            pmhdl.Free();
            sbxhdl.Free();
            dehdl.Free();
            umhdl.Free();
            spxhdl.Free();
            pcxhdl.Free();
            undxhdl.Free();

            return new Result(result, statistics);
        }

        /// <summary>Sets the seed used by the random number generator.</summary>
        /// <param name="value">The seed.</param>
        public static void SetSeed(uint value)
        {
            BorgLibrary.BORG_Random_seed(value);
        }

        /// <summary>Enable debugging output.</summary>
        public static void EnableDebugging()
        {
            BorgLibrary.BORG_Debug_on();
        }

        /// <summary>Disable debugging output.</summary>
        public static void DisableDebugging()
        {
            BorgLibrary.BORG_Debug_off();
        }
    }
	
	/// <summary>Helper functions for defining constraints.</summary>
	/// <remarks>
	/// These functions ensure several conditions hold.  First, if the constraint is satisfied, the value is 0.  If the constraint is
	/// violated, then the value is non-zero and will scale linearly with the degree of violation.
	/// </remarks>
	public class Constraint {

        /// <summary>Offset applied to constraint violations to ensure they are non-zero.</summary>
		public static double PRECISION = 0.1;

		private Constraint() {
			// do nothing
		}

		/// <summary>Defines the constraint x > y.</summary>
		public static double greaterThan(double x, double y, double epsilon = 0.0) {
			return x > y-epsilon ? 0.0 : y-x+PRECISION;
		}

		/// <summary>Defines the constraint x < y.</summary>
		public static double lessThan(double x, double y, double epsilon = 0.0) {
			return x < y+epsilon ? 0.0 : x-y+PRECISION;
		}

		/// <summary>Defines the constraint x >= y.</summary>
		public static double greaterThanOrEqual(double x, double y, double epsilon = 0.0) {
			return x >= y-epsilon ? 0.0 : y-x+PRECISION;
		}

		/// <summary>Defines the constraint x <= y.</summary>
		public static double lessThanOrEqual(double x, double y, double epsilon = 0.0) {
			return x <= y+epsilon ? 0.0 : x-y+PRECISION;
		}

		/// <summary>Defines the constraint x == y.</summary>
		public static double equal(double x, double y, double epsilon = 0.0) {
			return Math.Abs(y-x) < epsilon ? 0.0 : Math.Abs(y-x)+PRECISION;
		}

		/// <summary>Defines the constraint x == 0.</summary>
		public static double zero(double x, double epsilon = 0.0) {
			return equal(x, 0.0, epsilon);
		}

		/// <summary>Defines the constraint x >= 0.</summary>
		public static double nonNegative(double x, double epsilon = 0.0) {
			return greaterThanOrEqual(x, 0.0, epsilon);
		}
		
		/// <summary>Defines the constraint x > 0.</summary>
		public static double positive(double x, double epsilon = 0.0) {
			return greaterThan(x, 0.0, epsilon);
		}
		
		/// <summary>Defines the constraint x < 0.</summary>
		public static double negative(double x, double epsilon = 0.0) {
			return lessThan(x, 0.0, epsilon);
		}

		/// <summary>Requires all conditions to be satisfied.</summary>
		public static double all(params double[] args) {
			double result = 0.0;
			
			for (int i = 0; i < args.Length; i++) {
				result += args[i];
			}
			
			return result;
		}
	
		/// <summary>Requres at least one condition to be satisfied.</summary>
		public static double any(params double[] args) {
			double result = 0.0;
			
			for (int i = 0; i < args.Length; i++) {
				if (i == 0.0) {
					return 0.0;
				} else {
					result += args[i];
				}
			}
			
			return result;
		}
	
	}
	
	/// <summary>Function delegate used to define a multiobjective problem.</summary>
	/// <remarks>
	/// Any function with this method signature can be optimized using the Borg MOEA.  The optimization
	/// algorithm calls this method, supplying the decision variables for the current design being
	/// evaluated.  This body of this method evaluates those inputs and assigns the objective and
	/// constraint values, which are sent back to the optimization algorithm.
	/// </remarks>
	/// <param name="variables">The decision variables (input).</param>
	/// <param name="objectives">The objective values (output).</param>
	/// <param name="constraints">The constraint values (output).<param>
	public delegate void ObjectiveFunction(double[] variables, double[] objectives, double[] constriants);
	
    /// <summary>Wraps an ObjectiveFunction to provide a method compatible with the Borg MOEA API.</summary>
	class FunctionWrapper
	{

        /// <summary>The number of decision variables.</summary>
		private int numberOfVariables;

        /// <summary>The number of objectives.</summary>
		private int numberOfObjectives;

        /// <summary>The number of constraints.</summary>
		private int numberOfConstraints;

        /// <summary>The C# objective function.</summary>
		private ObjectiveFunction function;

        /// <summary>Wraps an ObjectiveFunction to provide a method compatible with the Borg MOEA API.</summary>
        /// <param name="numberOfVariables">The number of decision variables.</param>
        /// <param name="numberOfObjectives">The number of objectives.</param>
        /// <param name="numberOfConstraints">The number of constraints.</param>
        /// <param name="function">The C# objective function.</param>
		public FunctionWrapper(int numberOfVariables, int numberOfObjectives, int numberOfConstraints, ObjectiveFunction function)
		{
			this.numberOfVariables = numberOfVariables;
			this.numberOfObjectives = numberOfObjectives;
			this.numberOfConstraints = numberOfConstraints;
			this.function = function;
		}

        /// <summary>The evaluate method that is compatible with the Borg MOEA API.</summary>
        /// <param name="variablesPtr">Pointer to a double array storing the decision variables.</param>
        /// <param name="objectivesPtr">Pointer to a double array storing the objectives.</param>
        /// <param name="constraintsPtr">Pointer to a double array storing the constraint values.</param>
		public void Evaluate(IntPtr variablesPtr, IntPtr objectivesPtr, IntPtr constraintsPtr)
		{
			double[] variables = new double[numberOfVariables];
			double[] objectives = new double[numberOfObjectives];
			double[] constraints = new double[numberOfConstraints];

			Marshal.Copy(variablesPtr, variables, 0, numberOfVariables);
			
			function(variables, objectives, constraints);
			
			Marshal.Copy(objectives, 0, objectivesPtr, numberOfObjectives);
			
			if (numberOfConstraints > 0) {
				Marshal.Copy(constraints, 0, constraintsPtr, numberOfConstraints);
			}
		}
		
	}

    /// <summary>Function delegate for defining objective functions within the native Borg MOEA API</summary>
    /// <param name="variables">Pointer to a double array storing the decision variables.</param>
    /// <param name="objectives">Pointer to a double array storing the objectives.</param>
    /// <param name="constraints">Pointer to a double array storing the constraint values.</param>
    delegate void NativeObjectiveFunction(IntPtr variables, IntPtr objectives, IntPtr constriants);

    /// <summary>Function delegate for defining a dominance function.</summary>
    /// <param name="solution1">Pointer to the first solution (BORG_Solution).</param>
    /// <param name="solution2">Pointer to the second solution (BORG_Solution).</param>
    /// <returns>Flag indicating if solution1 is dominated by, non-dominated, or dominates solution2.</returns>
    delegate BORG_Dominance DominanceFunction(IntPtr solution1, IntPtr solution2);

    /// <summary>Function delegate for defining an operator.</summary>
    /// <param name="op">Pointer to the structure storing operator parameters (BORG_Operator).</param>
    /// <param name="parents">Pointer to an array of solutions used as parents (BORG_Solution*).</param>
    /// <param name="offspring">Pointer to an (initially empty) array of solutions where the offspring are returned (BORG_Solution*).</param>
    delegate void OperatorFunction(IntPtr op, ref IntPtr parents, ref IntPtr offspring);

    /// <summary>Static methods for directly calling the native Borg MOEA API.</summary>
    /// <remarks>
    /// The static methods contained within this class invoke the native Borg MOEA API directly.  Use of these methods should
    /// be avoided.  Use the Borg, Result, Solution, and ObjectiveFunction types instead.
    /// </remarks>
    partial class BorgLibrary
    {

        /// Return Type: void
        ///ptr: void*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Validate_pointer")]
        public static extern void BORG_Validate_pointer(IntPtr ptr);


        /// Return Type: void
        ///index: int
        ///size: int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Validate_index")]
        public static extern void BORG_Validate_index(int index, int size);


        /// Return Type: void
        ///ptr: void*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Validate_malloc")]
        public static extern void BORG_Validate_malloc(IntPtr ptr);


        /// Return Type: void
        ///value: double
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Validate_positive")]
        public static extern void BORG_Validate_positive(double value);


        /// Return Type: void
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Debug_on")]
        public static extern void BORG_Debug_on();


        /// Return Type: void
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Debug_off")]
        public static extern void BORG_Debug_off();


        /// Return Type: void
        ///name: char*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Debug_set_name")]
        public static extern void BORG_Debug_set_name(IntPtr name);


        /// Return Type: BORG_Problem->BORG_Problem_t*
        ///numberOfVariables: int
        ///numberOfObjectives: int
        ///numberOfConstraints: int
        ///function: Anonymous_ad9983d9_dbe9_4824_8e72_1d06c0fe3250
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Problem_create")]
        public static extern IntPtr BORG_Problem_create(int numberOfVariables, int numberOfObjectives, int numberOfConstraints, IntPtr function);


        /// Return Type: void
        ///problem: BORG_Problem->BORG_Problem_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Problem_destroy")]
        public static extern void BORG_Problem_destroy(IntPtr problem);


        /// Return Type: void
        ///problem: BORG_Problem->BORG_Problem_t*
        ///index: int
        ///epsilon: double
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Problem_set_epsilon")]
        public static extern void BORG_Problem_set_epsilon(IntPtr problem, int index, double epsilon);


        /// Return Type: void
        ///problem: BORG_Problem->BORG_Problem_t*
        ///index: int
        ///name: char*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Problem_set_name")]
        public static extern void BORG_Problem_set_name(IntPtr problem, int index, [System.Runtime.InteropServices.InAttribute()] [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.LPStr)] string name);


        /// Return Type: void
        ///problem: BORG_Problem->BORG_Problem_t*
        ///epsilons: double*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Problem_set_epsilons")]
        public static extern void BORG_Problem_set_epsilons(IntPtr problem, ref double epsilons);


        /// Return Type: void
        ///problem: BORG_Problem->BORG_Problem_t*
        ///index: int
        ///lowerBound: double
        ///upperBound: double
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Problem_set_bounds")]
        public static extern void BORG_Problem_set_bounds(IntPtr problem, int index, double lowerBound, double upperBound);


        /// Return Type: int
        ///problem: BORG_Problem->BORG_Problem_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Problem_number_of_variables")]
        public static extern int BORG_Problem_number_of_variables(IntPtr problem);


        /// Return Type: int
        ///problem: BORG_Problem->BORG_Problem_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Problem_number_of_objectives")]
        public static extern int BORG_Problem_number_of_objectives(IntPtr problem);


        /// Return Type: int
        ///problem: BORG_Problem->BORG_Problem_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Problem_number_of_constraints")]
        public static extern int BORG_Problem_number_of_constraints(IntPtr problem);


        /// Return Type: BORG_Solution->BORG_Solution_t*
        ///problem: BORG_Problem->BORG_Problem_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Solution_create")]
        public static extern IntPtr BORG_Solution_create(IntPtr problem);


        /// Return Type: void
        ///solution: BORG_Solution->BORG_Solution_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Solution_destroy")]
        public static extern void BORG_Solution_destroy(IntPtr solution);


        /// Return Type: BORG_Solution->BORG_Solution_t*
        ///original: BORG_Solution->BORG_Solution_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Solution_clone")]
        public static extern IntPtr BORG_Solution_clone(IntPtr original);


        /// Return Type: double
        ///solution: BORG_Solution->BORG_Solution_t*
        ///index: int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Solution_get_variable")]
        public static extern double BORG_Solution_get_variable(IntPtr solution, int index);


        /// Return Type: double
        ///solution: BORG_Solution->BORG_Solution_t*
        ///index: int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Solution_get_objective")]
        public static extern double BORG_Solution_get_objective(IntPtr solution, int index);


        /// Return Type: double
        ///solution: BORG_Solution->BORG_Solution_t*
        ///index: int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Solution_get_constraint")]
        public static extern double BORG_Solution_get_constraint(IntPtr solution, int index);


        /// Return Type: void
        ///solution: BORG_Solution->BORG_Solution_t*
        ///index: int
        ///value: double
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Solution_set_variable")]
        public static extern void BORG_Solution_set_variable(IntPtr solution, int index, double value);


        /// Return Type: void
        ///solution: BORG_Solution->BORG_Solution_t*
        ///variables: double*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Solution_set_variables")]
        public static extern void BORG_Solution_set_variables(IntPtr solution, ref double variables);


        /// Return Type: BORG_Problem->BORG_Problem_t*
        ///solution: BORG_Solution->BORG_Solution_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Solution_get_problem")]
        public static extern IntPtr BORG_Solution_get_problem(IntPtr solution);


        /// Return Type: void
        ///solution: BORG_Solution->BORG_Solution_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Solution_evaluate")]
        public static extern void BORG_Solution_evaluate(IntPtr solution);


        /// Return Type: void
        ///solution: BORG_Solution->BORG_Solution_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Solution_initialize")]
        public static extern void BORG_Solution_initialize(IntPtr solution);


        /// Return Type: int
        ///solution: BORG_Solution->BORG_Solution_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Solution_violates_constraints")]
        public static extern int BORG_Solution_violates_constraints(IntPtr solution);


        /// Return Type: BORG_Dominance
        ///solution1: BORG_Solution->BORG_Solution_t*
        ///solution2: BORG_Solution->BORG_Solution_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Dominance_pareto")]
        public static extern BORG_Dominance BORG_Dominance_pareto(IntPtr solution1, IntPtr solution2);


        /// Return Type: BORG_Dominance
        ///solution1: BORG_Solution->BORG_Solution_t*
        ///solution2: BORG_Solution->BORG_Solution_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Dominance_epsilon")]
        public static extern BORG_Dominance BORG_Dominance_epsilon(IntPtr solution1, IntPtr solution2);


        /// Return Type: BORG_Dominance
        ///solution1: BORG_Solution->BORG_Solution_t*
        ///solution2: BORG_Solution->BORG_Solution_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Dominance_constraints")]
        public static extern BORG_Dominance BORG_Dominance_constraints(IntPtr solution1, IntPtr solution2);


        /// Return Type: BORG_Dominance
        ///solution1: BORG_Solution->BORG_Solution_t*
        ///solution2: BORG_Solution->BORG_Solution_t*
        ///comparator1: DominanceFunction
        ///comparator2: DominanceFunction
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Dominance_compound")]
        public static extern BORG_Dominance BORG_Dominance_compound(IntPtr solution1, IntPtr solution2, DominanceFunction comparator1, DominanceFunction comparator2);


        /// Return Type: void
        ///seed: unsigned int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Random_seed")]
        public static extern void BORG_Random_seed(uint seed);


        /// Return Type: double
        ///lowerBound: double
        ///upperBound: double
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Random_uniform")]
        public static extern double BORG_Random_uniform(double lowerBound, double upperBound);


        /// Return Type: int
        ///n: int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Random_int")]
        public static extern int BORG_Random_int(int n);


        /// Return Type: double
        ///mean: double
        ///stdev: double
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Random_gaussian")]
        public static extern double BORG_Random_gaussian(double mean, double stdev);


        /// Return Type: BORG_Operator->BORG_Operator_t*
        ///name: char*
        ///numberOfParents: int
        ///numberOfOffspring: int
        ///numberOfParameters: int
        ///function: Anonymous_f8399d0b_0ecc_480b_982b_38087df5599f
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Operator_create")]
        public static extern IntPtr BORG_Operator_create([System.Runtime.InteropServices.InAttribute()] [System.Runtime.InteropServices.MarshalAsAttribute(System.Runtime.InteropServices.UnmanagedType.LPStr)] string name, int numberOfParents, int numberOfOffspring, int numberOfParameters, IntPtr function);


        /// Return Type: void
        ///variation: BORG_Operator->BORG_Operator_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Operator_destroy")]
        public static extern void BORG_Operator_destroy(IntPtr variation);


        /// Return Type: double
        ///variation: BORG_Operator->BORG_Operator_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Operator_get_probability")]
        public static extern double BORG_Operator_get_probability(IntPtr variation);


        /// Return Type: void
        ///variation: BORG_Operator->BORG_Operator_t*
        ///index: int
        ///parameter: double
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Operator_set_parameter")]
        public static extern void BORG_Operator_set_parameter(IntPtr variation, int index, double parameter);


        /// Return Type: void
        ///variation: BORG_Operator->BORG_Operator_t*
        ///parameters: double*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Operator_set_parameters")]
        public static extern void BORG_Operator_set_parameters(IntPtr variation, ref double parameters);


        /// Return Type: int
        ///variation: BORG_Operator->BORG_Operator_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Operator_get_number_of_offspring")]
        public static extern int BORG_Operator_get_number_of_offspring(IntPtr variation);


        /// Return Type: int
        ///variation: BORG_Operator->BORG_Operator_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Operator_get_number_of_parents")]
        public static extern int BORG_Operator_get_number_of_parents(IntPtr variation);


        /// Return Type: void
        ///variation: BORG_Operator->BORG_Operator_t*
        ///mutation: BORG_Operator->BORG_Operator_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Operator_set_mutation")]
        public static extern void BORG_Operator_set_mutation(IntPtr variation, IntPtr mutation);


        /// Return Type: void
        ///variation: BORG_Operator->BORG_Operator_t*
        ///parents: BORG_Solution*
        ///offspring: BORG_Solution*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Operator_apply")]
        public static extern void BORG_Operator_apply(IntPtr variation, ref IntPtr parents, ref IntPtr offspring);


        /// Return Type: void
        ///um: BORG_Operator->BORG_Operator_t*
        ///parents: BORG_Solution*
        ///offspring: BORG_Solution*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Operator_UM")]
        public static extern void BORG_Operator_UM(IntPtr um, ref IntPtr parents, ref IntPtr offspring);


        /// Return Type: void
        ///sbx: BORG_Operator->BORG_Operator_t*
        ///parents: BORG_Solution*
        ///offspring: BORG_Solution*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Operator_SBX")]
        public static extern void BORG_Operator_SBX(IntPtr sbx, ref IntPtr parents, ref IntPtr offspring);


        /// Return Type: void
        ///pm: BORG_Operator->BORG_Operator_t*
        ///parents: BORG_Solution*
        ///offspring: BORG_Solution*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Operator_PM")]
        public static extern void BORG_Operator_PM(IntPtr pm, ref IntPtr parents, ref IntPtr offspring);


        /// Return Type: void
        ///de: BORG_Operator->BORG_Operator_t*
        ///parents: BORG_Solution*
        ///offspring: BORG_Solution*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Operator_DE")]
        public static extern void BORG_Operator_DE(IntPtr de, ref IntPtr parents, ref IntPtr offspring);


        /// Return Type: void
        ///spx: BORG_Operator->BORG_Operator_t*
        ///parents: BORG_Solution*
        ///offspring: BORG_Solution*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Operator_SPX")]
        public static extern void BORG_Operator_SPX(IntPtr spx, ref IntPtr parents, ref IntPtr offspring);


        /// Return Type: void
        ///pcx: BORG_Operator->BORG_Operator_t*
        ///parents: BORG_Solution*
        ///offspring: BORG_Solution*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Operator_PCX")]
        public static extern void BORG_Operator_PCX(IntPtr pcx, ref IntPtr parents, ref IntPtr offspring);


        /// Return Type: void
        ///undx: BORG_Operator->BORG_Operator_t*
        ///parents: BORG_Solution*
        ///offspring: BORG_Solution*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Operator_UNDX")]
        public static extern void BORG_Operator_UNDX(IntPtr undx, ref IntPtr parents, ref IntPtr offspring);


        /// Return Type: BORG_Population->BORG_Population_t*
        ///capacity: int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Population_create")]
        public static extern IntPtr BORG_Population_create(int capacity);


        /// Return Type: void
        ///population: BORG_Population->BORG_Population_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Population_destroy")]
        public static extern void BORG_Population_destroy(IntPtr population);


        /// Return Type: void
        ///population: BORG_Population->BORG_Population_t*
        ///solution: BORG_Solution->BORG_Solution_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Population_add")]
        public static extern void BORG_Population_add(IntPtr population, IntPtr solution);


        /// Return Type: void
        ///population: BORG_Population->BORG_Population_t*
        ///newCapacity: int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Population_reset")]
        public static extern void BORG_Population_reset(IntPtr population, int newCapacity);


        /// Return Type: BORG_Solution->BORG_Solution_t*
        ///population: BORG_Population->BORG_Population_t*
        ///tournamentSize: int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Population_tournament")]
        public static extern IntPtr BORG_Population_tournament(IntPtr population, int tournamentSize);


        /// Return Type: void
        ///population: BORG_Population->BORG_Population_t*
        ///arity: int
        ///parents: BORG_Solution*
        ///tournamentSize: int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Population_select")]
        public static extern void BORG_Population_select(IntPtr population, int arity, ref IntPtr parents, int tournamentSize);


        /// Return Type: BORG_Archive->BORG_Archive_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Archive_create")]
        public static extern IntPtr BORG_Archive_create();


        /// Return Type: BORG_Archive->BORG_Archive_t*
        ///original: BORG_Archive->BORG_Archive_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Archive_clone")]
        public static extern IntPtr BORG_Archive_clone(IntPtr original);


        /// Return Type: void
        ///archive: BORG_Archive->BORG_Archive_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Archive_destroy")]
        public static extern void BORG_Archive_destroy(IntPtr archive);


        /// Return Type: void
        ///archive: BORG_Archive->BORG_Archive_t*
        ///solution: BORG_Solution->BORG_Solution_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Archive_add")]
        public static extern void BORG_Archive_add(IntPtr archive, IntPtr solution);


        /// Return Type: int
        ///archive: BORG_Archive->BORG_Archive_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Archive_get_size")]
        public static extern int BORG_Archive_get_size(IntPtr archive);


        /// Return Type: BORG_Solution->BORG_Solution_t*
        ///archive: BORG_Archive->BORG_Archive_t*
        ///index: int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Archive_get")]
        public static extern IntPtr BORG_Archive_get(IntPtr archive, int index);


        /// Return Type: int
        ///archive: BORG_Archive->BORG_Archive_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Archive_get_improvements")]
        public static extern int BORG_Archive_get_improvements(IntPtr archive);


        /// Return Type: BORG_Solution->BORG_Solution_t*
        ///archive: BORG_Archive->BORG_Archive_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Archive_select")]
        public static extern IntPtr BORG_Archive_select(IntPtr archive);


        /// Return Type: BORG_Algorithm->BORG_Algorithm_t*
        ///problem: BORG_Problem->BORG_Problem_t*
        ///numberOfOperators: int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_create")]
        public static extern IntPtr BORG_Algorithm_create(IntPtr problem, int numberOfOperators);


        /// Return Type: void
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_destroy")]
        public static extern void BORG_Algorithm_destroy(IntPtr algorithm);


        /// Return Type: void
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_validate")]
        public static extern void BORG_Algorithm_validate(IntPtr algorithm);


        /// Return Type: int
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_get_number_restarts")]
        public static extern int BORG_Algorithm_get_number_restarts(IntPtr algorithm);


        /// Return Type: int
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_get_number_improvements")]
        public static extern int BORG_Algorithm_get_number_improvements(IntPtr algorithm);


        /// Return Type: void
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        ///windowSize: int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_set_window_size")]
        public static extern void BORG_Algorithm_set_window_size(IntPtr algorithm, int windowSize);


        /// Return Type: void
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        ///maxWindowSize: int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_set_max_window_size")]
        public static extern void BORG_Algorithm_set_max_window_size(IntPtr algorithm, int maxWindowSize);


        /// Return Type: void
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        ///initialPopulationSize: int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_set_initial_population_size")]
        public static extern void BORG_Algorithm_set_initial_population_size(IntPtr algorithm, int initialPopulationSize);


        /// Return Type: void
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        ///minimumPopulationSize: int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_set_minimum_population_size")]
        public static extern void BORG_Algorithm_set_minimum_population_size(IntPtr algorithm, int minimumPopulationSize);


        /// Return Type: void
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        ///maximumPopulationSize: int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_set_maximum_population_size")]
        public static extern void BORG_Algorithm_set_maximum_population_size(IntPtr algorithm, int maximumPopulationSize);


        /// Return Type: void
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        ///populationRatio: double
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_set_population_ratio")]
        public static extern void BORG_Algorithm_set_population_ratio(IntPtr algorithm, double populationRatio);


        /// Return Type: void
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        ///selectionRatio: double
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_set_selection_ratio")]
        public static extern void BORG_Algorithm_set_selection_ratio(IntPtr algorithm, double selectionRatio);


        /// Return Type: void
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        ///restartMode: BORG_Restart
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_set_restart_mode")]
        public static extern void BORG_Algorithm_set_restart_mode(IntPtr algorithm, BORG_Restart restartMode);


        /// Return Type: void
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        ///probilityMode: BORG_Probabilities
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_set_probability_mode")]
        public static extern void BORG_Algorithm_set_probability_mode(IntPtr algorithm, BORG_Probabilities probilityMode);


        /// Return Type: int
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_get_mutation_index")]
        public static extern int BORG_Algorithm_get_mutation_index(IntPtr algorithm);


        /// Return Type: void
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        ///maxMutationIndex: int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_set_max_mutation_index")]
        public static extern void BORG_Algorithm_set_max_mutation_index(IntPtr algorithm, int maxMutationIndex);


        /// Return Type: void
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        ///index: int
        ///variation: BORG_Operator->BORG_Operator_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_set_operator")]
        public static extern void BORG_Algorithm_set_operator(IntPtr algorithm, int index, IntPtr variation);


        /// Return Type: void
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_update")]
        public static extern void BORG_Algorithm_update(IntPtr algorithm);


        /// Return Type: int
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_select")]
        public static extern int BORG_Algorithm_select(IntPtr algorithm);


        /// Return Type: void
        ///numberOfParents: int
        ///parents: BORG_Solution*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_shuffle")]
        public static extern void BORG_Algorithm_shuffle(int numberOfParents, ref IntPtr parents);


        /// Return Type: void
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_restart")]
        public static extern void BORG_Algorithm_restart(IntPtr algorithm);


        /// Return Type: int
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_check")]
        public static extern int BORG_Algorithm_check(IntPtr algorithm);


        /// Return Type: void
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_step")]
        public static extern void BORG_Algorithm_step(IntPtr algorithm);


        /// Return Type: int
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_get_nfe")]
        public static extern int BORG_Algorithm_get_nfe(IntPtr algorithm);


        /// Return Type: int
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_get_population_size")]
        public static extern int BORG_Algorithm_get_population_size(IntPtr algorithm);


        /// Return Type: int
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_get_archive_size")]
        public static extern int BORG_Algorithm_get_archive_size(IntPtr algorithm);


        /// Return Type: BORG_Archive->BORG_Archive_t*
        ///algorithm: BORG_Algorithm->BORG_Algorithm_t*
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_get_result")]
        public static extern IntPtr BORG_Algorithm_get_result(IntPtr algorithm);


        /// Return Type: BORG_Archive->BORG_Archive_t*
        ///problem: BORG_Problem->BORG_Problem_t*
        ///maxEvaluations: int
        [System.Runtime.InteropServices.DllImportAttribute("borg.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "BORG_Algorithm_run")]
        public static extern IntPtr BORG_Algorithm_run(IntPtr problem, int maxEvaluations);

    }

}
