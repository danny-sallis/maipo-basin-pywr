using System;
using System.Runtime.InteropServices;
using BorgMOEA;

namespace Example
{
    class Example
    {
        public static int nvars = 11;
        public static int nobjs = 2;

        public static void DTLZ2(double[] vars, double[] objs, double[] constrs)
        {
            int k = nvars - nobjs + 1;
            double g = 0.0;

            for (int i = nvars - k; i < nvars; i++)
            {
                g += Math.Pow(vars[i] - 0.5, 2.0);
            }

            for (int i = 0; i < nobjs; i++)
            {
                objs[i] = 1.0 + g;

                for (int j = 0; j < nobjs - i - 1; j++)
                {
                    objs[i] *= Math.Cos(0.5 * Math.PI * vars[j]);
                }

                if (i != 0)
                {
                    objs[i] *= Math.Sin(0.5 * Math.PI * vars[nobjs - i - 1]);
                }
            }
        }

        static void Main(string[] args)
        {
            Borg borg = new Borg(nvars, nobjs, 0, DTLZ2);

            for (int i = 0; i < nvars; i++)
            {
                borg.SetBounds(i, 0.0, 1.0);
            }

            for (int i = 0; i < nobjs; i++)
            {
                borg.SetEpsilon(i, 0.01);
            }

            Result result = borg.Solve(1000000);

            foreach (Solution solution in result)
            {
                Console.WriteLine(solution.GetObjective(0) + " " + solution.GetObjective(1));
            }

            if (!Console.IsInputRedirected && Console.KeyAvailable)
            {
                Console.WriteLine();
                Console.WriteLine("Press any key to exit...");
                Console.ReadKey();
            }
        }
    }
}
