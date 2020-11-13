using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace gauss_
{
    class gauss_
    {
        public static int GaussMethod(int m, int n, double[,] a, double eps)
        {
            
            int k;
            int l = 0;
            double r; //для записи элемента матрицы(при поиске разрешающего элемента)
            int i = 0;
            int j = 0;


            while (i < m && j < n)
            {
                //поиск максимального элемента в j-том столб., начиная с i-той строки
                r = 0.0;
                for (k = i; k < m; ++k)
                {
                    if (Math.Abs(a[k,j]) > r)
                    {
                        l = k; //Запомним номер строки
                        r = Math.Abs(a[k,j]); //и максимальный элемент
                    }
                }

                if (r <= eps)//столбец 0
                {
                    //Все элементы j-го столбца по абсолютной величине
                    //не првеосходят eps.
                    //Обнулим столбце, начиная с j-й строки
                    for (k = i; k < m; ++k)
                    {
                        a[k, j] = 0.0;
                    }
                    ++j; // Увеличим индекс столбца
                    continue; //Переходим к следующей итерации
                }
                if (l != i)
                {
                    //Меняем местами i-ю и l-ю строки
                    for (k = j; k < n; ++k)
                    {
                        r = a[i,k];
                        a[i,k] = a[l,k];
                        a[l, k] = (-r); //Меняем знак строки
                    }
                }
                //Утверждение: fabs(a[i,j]) > eps
                r = a[i,j];
                if (Math.Abs(r) > eps)
                {
                    Console.Write("all is ok\n");
                }
                else
                {
                    Console.Write("eror\n");
                }
                //assert(fabs(r) > eps);

                //Обнуляем j-й столбец, начиная со строки i+1,
                //применяя элементарные преобразования 2-го рода
                for (k = i + 1; k < m; ++k)
                {
                    double c = (-a[k,j]) / r;
                    //К k-й строке прибавляем i-ю,
                    //умноженную на c
                    a[k,j] = 0.0;
                    for (l = j + 1; l < n; ++l)
                    {
                        a[k,l] += c * a[i,l];
                    }
                }

                ++i;
                ++j; //Переходим к следующему минору
            }
            return i; //Возвращаем количество ненулевых строк
        }
        
        static void Main(string[] args)
        {
            Console.Write("Введите размеры матрицы m, n: ");

            int m = int.Parse(Console.ReadLine()); //  кол-во строк
            int n = int.Parse(Console.ReadLine()); // кол-во столбцов
            double[,] a = new double[m, n]; //расширенная матрица
            for(int i = 0; i < m; i++) //заполнение от пользователя
                for(int j = 0; j < n; j++)
                {
                    Console.Write($"Введите элемент с индексом {i}_{j}: ");
                    a[i, j] = int.Parse(Console.ReadLine());
                }

            Console.Write("Введите точность вычислений eps: ");
            double eps = double.Parse(Console.ReadLine());

            //Вызываем метод Гаусса
            int rank = GaussMethod(m, n, a, eps);

            //Печатаем ступенчатую матрицу
            Console.Write("Ступенчатый вид матрицы: \n");
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    Console.Write("{0,10:f3} ", a[i, j]);
                }
                Console.WriteLine(); //Перевести строку

            }
                
            
            //Печатаем ранг матрицы
            Console.Write("Ранг матрицы = {0:D}\n", rank);

            if(m == n) //Для квадратной матрицы вычисляем и печатаем ее определитель;
            {
                
                double det = 1.0;
                for (int i = 0; i < m; i++)
                {
                    for(int j = 0; j < m; j++)
                        if (j == i)
                            det *= a[i, j];
                 
                }
                Console.Write("Определитель матрицы = {0:f3}\n", det);
            }

            //обратный ход
            int count = Convert.ToInt32(a.GetLongLength(0));
            double[] X = new double[count];
            double[,] A = new double[a.GetLongLength(0), a.GetLongLength(1)-1];
            double[] B = new double[a.GetLongLength(0)];
            for (int i = 0; i < A.GetLongLength(0); i++)
            {
                for (int j = 0; j < A.GetLongLength(1); j++)
                    A[i, j] = a[i, j];
            }
            for (int i = 0; i < B.Length; i++)
                B[i] = a[i, a.GetLongLength(1) - 1];
            // решение системы: обратный ход

            X[count - 1] = B[count - 1] / A[count - 1, count - 1];
            for (int j = count - 2; j >= 0; j--)
            {
                double sum = 0;
                for (int k = count - 1; k >= j; k--)
                {
                    sum += A[j, k] * X[k];
                }
                X[j] = (B[j] - sum) / A[j, j];
            }
            a = null; //Освобождаем память
            for (int i = 0; i < X.Length; i++)
                Console.WriteLine($"x{i} = {X[i]}");
            Console.ReadKey();
        }
    }
}
