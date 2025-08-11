setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

x <- read.csv('ed_sample.csv')

effective_dims <- estimate.ED(x)

matrix <- cor(x)
eigen_val = sort(eigen(matrix)$values)

K = length(eigen_val)	
eigen_sum = sum(eigen_val)
norm_eigen_val = eigen_val/eigen_sum
eigen_var = var(eigen_val)*((K-1)/K)

n1 = prod(norm_eigen_val^(-norm_eigen_val))  
n2 = (eigen_sum^2)/sum(eigen_val^2) 
nInf = eigen_sum/max(eigen_val)    
nC = K - ((K^2)/(eigen_sum^2))*eigen_var  

print(list(n1, n2, nInf, nC))
      