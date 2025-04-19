data(iris)
library(MASS)

set.seed(1)
train_indices <- sample(1:nrow(iris), 0.7*nrow(iris))
train <- iris[train_indices,]
test <- iris[-train_indices,]

#
y_train <- train[,5]
X_train <- as.matrix(train[,-5])

categories <- levels(y_train)
K <- length(categories)
n <- nrow(train)
p <- 4

# Create Y matrix where the element at row i and column j is 1 if the i'th
# observation is a member of the j'th category and 0 else wise.
Y_train <- matrix(0,ncol=K,nrow=n)
for (c in 1:K) {
  Y_train[,c] <- as.integer(y_train==categories[c])
}

# initial parameter estimate
b_hat <- matrix(0,ncol=K,nrow=p)

softmax <- function(Z) {
  Z <- exp(Z);
  Z/sum(Z)
}

get_probs_mat <- function(beta, X,k,n) {
  probs_mat <- matrix(0, ncol=k, nrow=n)
  for (i in 1:n) {
    probs_mat[i,] <- softmax(t(beta)%*%(X[i,]))
  }
  probs_mat
}

get_hessian <- function(X,probs_vec) {
  W <- diag(probs_vec*(1-probs_vec))
  -t(X)%*%W%*%X
}

get_update_matrix <- function(X,Y,probs_mat,K,p) {
  update_mat <- matrix(0,ncol=K,nrow=p)
  for (c in 1:(K-1)) {
    probs_vec_c <- probs_mat[,c]
    Y_c <- Y[,c]
    H_c <- get_hessian(X,probs_vec_c)
    print(H_c)
    update_mat[,c] <- -ginv(H_c)%*%t(X)%*%(Y_c-probs_vec_c)
  }
  update_mat
}

solve <- function(beta_start, X, Y, K, n, p) {
  iter = 0
  beta <- beta_start
  while (TRUE) {
    
    probs_mat <- get_probs_mat(beta,X,K,n)
    update_matrix <- get_update_matrix(X,Y,probs_mat,K,p)
    beta <- beta + update_matrix
    change <- sum(update_matrix^2)

    if (iter > 100) {
      break
    }
    if (change <= 0.01) {
      break
    }
    iter <- iter + 1
  }
  beta
}

b_hat <- solve(b_hat, X_train, Y_train, K, n, p)

X_test <- as.matrix(test[,-5])
y_test <- test[,5]
Y_test <- matrix(0,ncol=K,nrow=nrow(X_test))
for (c in 1:K) {
  Y_test[,c] <- as.integer(y_test==categories[c])
}

predicted_probs <- get_probs_mat(b_hat, X_test, K, nrow(X_test))
predictions <- apply(predicted_probs, 1, which.max)
error <- sum(apply(Y_test, 1, which.max)!=predictions)/nrow(X_test)
print(error)
