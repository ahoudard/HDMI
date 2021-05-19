import torch
from torch import nn

class HDMI():
    def __init__(self, patches, n_groups, sigma, device):
        super(HDMI, self).__init__()
        n_patch, dim_patch = patches.shape
        self.mu = torch.rand(dim_patch, n_groups, device=device)
        self.d = torch.ones(n_groups, device=device)
        self.Q = [torch.zeros(dim_patch, 1, device=device) for k in range(n_groups)]
        self.pi = torch.ones(n_groups, device=device)/n_groups
        self.ev = torch.ones(n_groups, dim_patch, device=device)        
        self.b = torch.tensor(sigma, device=device)
        self.y = patches
        self.p = dim_patch
        self.K = n_groups
        self.t = torch.zeros(n_groups, n_patch, device=device)
        self.eps = torch.tensor(torch.finfo(torch.float32).eps, device=device)

    def E_step(self):
        logt = torch.zeros_like(self.t)
        for k in range(self.K):
            idx = int(self.p-self.d[k])
            evk = self.ev[k, idx:]
            iev = torch.sqrt(torch.abs(1/evk - 1/self.b))
            muk = self.mu[:,k]
            A = torch.matmul(self.y, self.Q[k]*iev) - torch.matmul(muk, self.Q[k]*iev)
            logt[k,:] = -torch.sum(A**2, 1) + torch.sum(torch.log(evk)) + idx*torch.log(self.b) - 2*torch.log(self.pi[k]) - (2*torch.matmul(self.y,muk)- torch.matmul(muk, muk))/self.b # 
        for k in range(self.K):
            self.t[k,:] = 1/torch.sum(torch.exp((logt[k,:].unsqueeze(1)-logt.transpose(1,0))/2), 1)

    def M_step(self):
        self.pi = torch.mean(self.t,1)
        for k in range(self.K):
            tk = self.t[k,:]
            stack = self.y[tk>self.eps,:]
            tk = tk[tk>self.eps]
            tk = tk/torch.sum(tk)
            self.mu[:,k] = torch.mean(tk.unsqueeze(1)*stack, 0)
            centeredstack = stack - self.mu[:,k]
            Sk = torch.matmul(centeredstack.transpose(1,0), tk.unsqueeze(1)*centeredstack)
            ev, v = torch.symeig(Sk, eigenvectors=True)
            meanev = torch.cumsum(ev,0)/torch.arange(1,self.p+1, device=self.ev.device)
            self.d[k] = torch.sum((meanev-self.b)>0)
            self.ev[k,:] = ev
            self.Q[k] = v[:,int(self.p-self.d[k]):]
      
    def denoise(self):
        patches = self.y
        denoised_patches = torch.zeros_like(patches)
        for k in range(self.K):
            tk = self.t[k,:]
            stack = patches[tk>self.eps,:]
            tk = tk[tk>self.eps]
            Qk = self.Q[k]
            idx = int(self.p-self.d[k])
            iDelta = (self.b - self.ev[k, idx:])/(self.ev[k, idx:])
            centeredstack = stack - self.mu[:,k]    
            Mk = torch.eye(self.p, device=Qk.device) + torch.matmul(Qk*iDelta.unsqueeze(0),Qk.transpose(1,0))
            product = torch.matmul(Mk, centeredstack.transpose(1,0))
            denoised_patches[self.t[k,:]>self.eps,:] = denoised_patches[self.t[k,:]>self.eps,:] + tk.unsqueeze(1)*(stack - product.transpose(1,0))                
        return denoised_patches