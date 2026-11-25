import numpy as np
import plt
from scipy.sparse.linalg import svds
import pandas as pd

#task 1
# def svd(A):
#     A = np.array(A, dtype=float)
#     m, n = A.shape  # rows and columns
#     eigvals, V = np.linalg.eigh(A.T @ A)
#     order = np.argsort(eigvals)[::-1]
#     eigvals = eigvals[order]
#     V = V[:, order]
#     sing = [np.sqrt(eigval) for eigval in eigvals if eigval > 1e-10] #sometimes when we calculate 0, computer calculates really small numbers because of noises so we need to exclude them
#     sigma = np.zeros((m,n))
#     for i in range(len(sing)):
#         sigma[i][i] = sing[i]
#     U = np.zeros((m, m))
#     eigvals_additional, U_additional = np.linalg.eigh(A @ A.T)
#     order_additional = np.argsort(eigvals_additional)[::-1]
#     eigvals_additional = eigvals_additional[order_additional]
#     U_additional = U_additional[:, order_additional]
#     for i in range(m):
#         if i < len(sing):
#             U[:, i] = A @ V[:, i] / sing[i]  # hint
#         else:
#             U[:, i] = U_additional[:, i]  # for not implementing Gramm-Schmidt algorithm here, I just found eigenvectors of U and when lacks i just put them sorted in final U matrix
#     return U, sigma, V.T
#
# A = [[2,0,0],[0,3,0]]
# U, sigma, V = svd(A)
# print(U)
# print(sigma)
# print(V)
# print(U @ sigma @ V)
#task 2
#part 1
# file_path = 'ratings.csv'
# df = pd.read_csv(file_path)
# ratings_matrix = df.pivot(index='userId', columns='movieId',values='rating')
# print(ratings_matrix.shape)
# print("Number of ratings:", len(df))
# print("Number of users:", df['userId'].nunique())
# print("Number of movies:", df['movieId'].nunique())
# ratings_matrix = ratings_matrix.dropna(thresh= 50, axis=0)   # users with >= 50 ratings
# ratings_matrix = ratings_matrix.dropna(thresh=30, axis=1)   # movies with >= 30 ratings
# print(ratings_matrix.shape)
# ratings_matrix_filled = ratings_matrix.fillna(2.5)
# R = ratings_matrix_filled.values #convert table to a NumPy array
# user_ratings_mean = np.mean(R, axis=1) #average rating of each user (row)
# R_demeaned = R - user_ratings_mean.reshape(-1, 1) #subtract each user's average from their ratings
# U, sigma, Vt = svds(R_demeaned, k=3) #k corresponds to the dimensionality of the data we preserve(compacted svd)
# print(U, sigma, Vt)  # U: users x 3, Vt: 3 x movies
# print(U.shape, sigma.shape, Vt.shape)  # U: users x 3, Vt: 3 x movies
#
# n = 20
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(U[:n, 0], U[:n, 1], U[:n, 2])
# ax.set_title('Users')
# plt.show()#the closer the points are, the more similar the tastes of these people
#
# k = 20
# fig2 = plt.figure()
# bx = fig2.add_subplot(projection='3d')
# bx.scatter(Vt.T[:k, 0], Vt.T[:k, 1], Vt.T[:k, 2])
# bx.set_title('Films')
# plt.show()#the closer the points are, the more similar are the films by rating

#part 2
pd.set_option('display.max_columns', None)  #show all columns
pd.set_option('display.width', 200) #make the output wider
df = pd.read_csv('ratings.csv')
ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')
ratings_matrix = ratings_matrix.dropna(thresh=20, axis=0)  #keep users with >= 20 ratings
ratings_matrix = ratings_matrix.dropna(thresh=20, axis=1)  #keep movies with >= 20 ratings
print(ratings_matrix.shape)
ratings_matrix_filled = ratings_matrix.fillna(2.5)
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)
known = ~ratings_matrix.isna().values  #True where a real rating existed
preds_df = None
def predict(k):
    U, sigma, Vt = svds(R_demeaned, k=k)
    sigma = np.diag(sigma)
    return np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
for k in [1, 5, 10, 15, 50, 100]:
    pred = predict(k)
    error = np.mean(np.abs(pred[known] - R[known]))
    if k == 50:
        all_user_predicted_ratings = predict(k)
        preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)
    print("k:", k, "error:", error)
print("So since k is bigger and we get real details, the error became smaller")
print(preds_df)
only_predicted = preds_df.where(ratings_matrix.isna())#no sense to suggest the films that were already seen
print(only_predicted)

movies = pd.read_csv('movies.csv')
def recommendation(user_id, n=10):
    user_preds = only_predicted.loc[user_id]
    top = user_preds.sort_values(ascending=False).head(n)
    result = pd.DataFrame({'movieId': top.index, 'predicted_rating': top.values.round(2)})
    result = result.merge(movies, on='movieId')  # add title and genres
    print(f"Prediction for user {user_id}:")
    print(result[['movieId', 'title', 'genres', 'predicted_rating']])

recommendation(1)
recommendation(127)
