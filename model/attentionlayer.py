# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttFlowLayer(nn.Module):
    def __init__(self, embed_length):
        super(AttFlowLayer,self).__init__()
        #self.batch_size = batch_size
        self.embed_length = embed_length
        self.alpha = nn.Linear(3*embed_length,1,bias=False)

    def forward(self, context, query):
        #Compute similarity matrix
        #context [N,T,2d]   query [J,2d]
        batch_size = context.shape[0]
        query = query.unsqueeze(0).expand((batch_size,query.shape[0], self.embed_length)) #[J,2d]->[N,J,2d]
        shape = (batch_size, context.shape[1], query.shape[1], self.embed_length)
        context_extended = context.unsqueeze(2).expand(shape)                       #[N,T,2d]->[N,T,1,2d]->[N,T,J,2d]
        query_extended = query.unsqueeze(1).expand(shape)              #[J,2d]->[N,1,J,2d]->[N,T,J,2d]
        multiplied = torch.mul(context_extended,query_extended)
        cated = torch.cat((context_extended,query_extended,multiplied),3)           #[h;u;h*u]
        S = self.alpha(cated).view(batch_size,context.shape[1],query.shape[1]) #[N,T,J,1] -> #[N,T,J]
        #print(S)
        #outputs = S / (self.embed_length** 0.5)

        S_softmax_row = F.softmax(S,dim=1).permute(0,2,1)             #[N,J,T]
        S_softmax_col = F.softmax(S,dim=2)              #[N,T,J]

        query_masks = torch.sign(torch.abs(torch.sum(query, dim=-1)))  # (N, J)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, context.size()[1])  # (N, J, T)
        S_softmax_row = S_softmax_row * query_masks  # (N, J, T)

        #Q to C Attention
        #[N,J,T,2d]
        S_softmax_row_1 = S_softmax_row.unsqueeze(3).expand(S_softmax_row.shape[0],S_softmax_row.shape[1],S_softmax_row.shape[2],self.embed_length)
        context_1 = context_extended.permute(0,2,1,3) #[N,J,T,2d]

        attd = torch.mul(S_softmax_row_1,context_1) #[N,J,T,2d]

        G = torch.sum(attd,1)#[N,T,2d]
        H = torch.sum(attd,2)#[N,J,2d]

        G = torch.cat((context,G),2)

        return G,H





