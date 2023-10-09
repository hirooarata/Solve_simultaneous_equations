# elimnation_solve.py
# CC0
# Solve Simultaneous Equations Using Elimination Method

# from __future__ import absolute_import
# from __future__ import print_function
import numpy as np
import pandas as pd
# from six.moves import range

# 2023/05/07 bug
# 2023/05/08 bug
# 2023/05/09 OK?
# 2023/05/10 May be Ok
# 2023/05/11 Add check
class EliminationSolve:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.n = 3
        self.n = self.a.shape[0]# サイズ
        self.k = 0  # iteration index = left column index
        self.c = pd.DataFrame(np.zeros((self.n, self.n + 1)))
        # working memory,c=[a;b;]
        self.x = pd.Series(np.zeros(self.n))#答え
        self.order = pd.Series(list(range(self.n)))
        # 順序ベクトル[0：n-1]=[0,1,2,...,n-1]
        self.max_value_column_index = 0
        self.matrix_singular = False
        self.matrix_rank = self.n
        return

    # 連立方程式ax=bの行列aと列ベクトルbを連結して行列ｃを作る
    def concatenate(self):
        # データ構造
        # a[ 0:n-1, 0:n-1 ]
        # b[ 0:n-1 ]
        # c[ 0:n-1, 0:n ]=[ a; b; ]
        for i in range(0, self.n):
            self.c.iloc[i, self.n] = self.b[i]
        for i in range(0, self.n):
            for j in range(0, self.n):
                self.c.iloc[i, j] = self.a.iloc[i, j]
        # print('concatenate():c\n', self.c)
        return self
        
		# 前進消去
		#
		# ピボットを選ぶ
    # データフレームc中の列ベクトルc[:,j]ごとの絶対値の最小値を求め、
    # これらを並べて絶対値最小を並べた行ベクトルを作り、
    # その中から絶対値最大のカラムを探す(下請け関数)
    def search_pivot_row_vector(self):
        # print('search_pivot_row_vector():search_area is:[k:n, k:n]:[', self.k, ':', self.n-1, ',', self.k,
        #      ':', self.n-1, "]")
        search_area = self.c.iloc[self.k : self.n, self.k : self.n]
        # print('search_area:\n',search_area)
        search_area_abs_vmin_hmax_index = (
            np.argmax((np.abs(search_area)).min(axis=0)) + self.k
        )
        # print('search_area_abs_vmin_hmax_index=',search_area_abs_vmin_hmax_index)
        return search_area_abs_vmin_hmax_index

    # 列の交換とインデックスの順番orderの交換(下請け関数)
    def swap_pivot_row_vector(self, _i, _j):
        temp = self.c[_i]  # matrix
        self.c[_i] = self.c[_j]
        self.c[_j] = temp
        #
        temp1 = self.order.iloc[_i]  # row vector
        self.order.iloc[_i] = self.order.iloc[_j]
        self.order.iloc[_j] = temp1
        # print('swap_pivot_row_vector():order', self.order)
        # print('swap_pivot_row_vector():c', self.c)
        return self

    # 絶対値の最小値が最も大きい列max_value_column_indexと、
    # 左端の列(=ピボット列=self.k)を交換する
    def search_pivot_row_vector_and_swap(self):
        # print('search_pivot_row_vector_and_swap():')
        max_value_column_index = self.search_pivot_row_vector()
        # ノルムが最大の列を選ぶ
        if max_value_column_index != self.k:
            self.swap_pivot_row_vector(
                self.k, max_value_column_index
            )  # 左端の列kとノルム最大の列を交換
        print("search_pivot_row_vector_and_swap():c\n", self.c)
        print("------------------------")
        return self

    #  aのk番目の値で、行ベクトルj=k+1..nを割る
    def divine_with_pivot_vector(self):
        # print("divine_with_pivot_vector")
        #  developed with MacBook Monterey, gcc for c99 required
        #  aのk番目の値で、行ベクトルj=k+1..nを割る
        #     for (int i = k; i < n; i++ ) {
        #        for (int j=k + 1; j < n+1; j++ ) {
        #            c[i][j] /= c[i][k];}} // i行k列の値で割る
        for i in range(self.k, self.n):  # 縦方向 i=k..n-1
            # c_min = self.c.iloc[:, i].abs().min(axis=0)
            ## print('\nc_min=', c_min, '\n')
            # if c_min < 1e-6:
            #    self.matrix_singular = True
            #    self.matrix_rank = self.k
            #    print('divine_with_pivot_vector:c_min', c_min, '< 1e-6', self.matrix_rank)  # swap_lines()
            #    return self
            # else:
            # for j in range(self.k + 1, self.n + 1):
            denominator = self.c.iloc[i, self.k]
            for j in range(self.n, self.k - 1, -1):
                self.c.iloc[i, j] /= denominator  # i行k列の値で割る
        print("divine_with_pivot_vector:\n", self.c)
        print("------------------------")
        self.matrix_sigular = False
        # print('divine_with_pivot_vector:c_min=', c_min)  # swap_lines()
        return self

    # ある行とその下の行の差分を行列下から上へ作る。
    # 一番下の行の絶対値ノルムが0ならマトリックスシンギュラーなので真を返す。
    def sub_lines(self):
        # for (int i = k;  i < n-1; i++ ) {
        #    for (int j=k + 1; j < n+1; j++ ) {
        #         c[i + 1][j] =  - c[i + 1][j] + c[i][j];}}
        # for i in range(self.k, self.n - 1): # 行
        for i in range(self.n - 1, self.k, -1):  # 行,下から上へ
            # print("sub_lines:i=", i, ",j=[", self.k, ":", self.n, "]")
            for j in range(self.k, self.n + 1):  # 列、左から右へ
                # print("sub_lines:i=", i, ", j=", j)
                self.c.iloc[i, j] = self.c.iloc[i - 1, j] - self.c.iloc[i, j]

        norm_h_vector_k = self.c.iloc[self.k + 1, self.k : self.n].abs().sum() / self.n
        print("sub_lines():c[k+1,k:n-1]:\n", self.c.iloc[self.k + 1, self.k : self.n])
        print("sub_lines():norm_h_vector_k=", norm_h_vector_k)
        if norm_h_vector_k <= 3e-14:
            self.matrix_singular = True
            self.matrix_rank = self.k + 1
            self.c.iloc[self.k + 1, self.n] = np.nan
            print("sub_lines():after sub:c:\n", self.c)
            print("sub_lines():matrix_singurlar:matrix_rank=", self.matrix_rank)
        print("------------------------")
        return self, self.matrix_singular

    # 後退解法
    def backward_solve(self, n_rank, n):
        print("---backward_solve:matrix:c.iloc[0:n_rank, 0:n_rank]:\n",
                self.c.iloc[0:n_rank, 0:n_rank],)
        # x[n - 1] = c[n - 1][n] / c[n - 1][n-1];
        # for (int i = n - 2;  i > -1; i-- ) {
        #    double sp = 0.0;
        #    for (int j = i;  j < n; j++ ) {
        #        sp += c[i][j] * x[j];  // c[i][0:k]*x[0:k]
        #    }
        #    x[i] = c[i][n] - sp;}
        if self.c.iloc[n_rank - 1, n_rank - 1] != 0:
            self.x[n_rank - 1] = (
                self.c.iloc[n_rank - 1, n] / self.c.iloc[n_rank - 1, n_rank - 1]
            )
        else:
            print("error:divided by zero in backward_solve!")
            self.x[n_rank - 1] = np.nan
        for i in range(n_rank - 2, -1, -1):
            sp = 0.0
            for j in range(i, n_rank, 1):
                sp += self.c.iloc[i, j] * self.x[j]  # self.c[i][0:k]*x[0:k]
            self.x[i] = self.c.iloc[i, n] - sp
        return self

        # 並べなおし

    def backward_reorder_x(self):
        temp = self.x
        if self.n > self.matrix_rank:
            for i in range(self.matrix_rank, self.n):
                self.x[i] = 0
        temp = self.x[self.order]
        self.x = temp
        return self

    # solve_linear main routine
    def solve_linear(self):
        #  Forward reduction
        self.concatenate()  # 行列aの右側にベクトルbを連結して行列cを作る
        # print("------------------------------------------")
        for self.k in range(0, self.n - 1):  # k = 0..n-2
            print("iteration k=", self.k)
            self.search_pivot_row_vector_and_swap()
            self.divine_with_pivot_vector()  # cのk番目の値で、行ベクトルj=k+1..nを割る
            if (
                self.sub_lines() == True
            ):  # c[i,k:n]=c[i,k:n]-c[i+1,k:n],i=0..n-1,k=0..n-1  # print('\nsub_lines\n', self.c)
                break  # matrix==0
        self.backward_solve(self.matrix_rank, self.n)  # Backward solve
        # print('pivot_vector():order:\n', self.order,'\n')
        self.backward_reorder_x()
        return self.x


def main():  # メイン
    a = pd.DataFrame([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0], [9.0, 10.0, 11.0]])
    b = pd.Series([4.0, 8.0, 12.0])
    s = EliminationSolve(a, b)
    print("---\na:\n", s.a, "\n---")
    print("b:\n", s.b, "\n---")
    x = s.solve_linear()
    print("x:\n", x, "\n---")


if __name__ == "__main__":
    main()
