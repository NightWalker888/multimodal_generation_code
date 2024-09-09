class AdaGroupNorm(nn.Module): 
 """ 
 修改 GroupNorm 层，以实现时间步编码信息的注入
 """ 
 def __init__( 
    self, embedding_dim: int, out_dim: int, num_groups: int, act_fn: 
    Optional[str] = None, eps: float = 1e-5 
    ): 
    super().__init__() 
    self.num_groups = num_groups 
    self.eps = eps 
    if act_fn is None: 
        self.act = None 
    else: 
        self.act = get_activation(act_fn) 
    self.linear = nn.Linear(embedding_dim, out_dim * 2) 

 def forward(self, x, emb): 
    ''' 
    x 是输入的潜在表示
    emb 是时间步编码
    ''' 
    if self.act: 
        emb = self.act(emb) 
        
    # DALL·E 3 中提到的
    # "a learned scale and bias term that 
    # is dependent on the timestep signal 
    # is applied to the outputs of the 
    # GroupNorm layers" 
    # 对应的就是下面这几行代码
    
    emb = self.linear(emb) 
    emb = emb[:, :, None, None] 
    scale, shift = emb.chunk(2, dim=1) 
    # F.group_norm 只减去均值再除以方差
    x = F.group_norm(x, self.num_groups, eps=self.eps) 
    # 使用根据时间步编码计算得到的缩放参数和偏移完成组归一化的缩放和偏移变换
    x = x * (1 + scale) + shift 
    return x