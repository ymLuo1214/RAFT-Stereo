import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    autocast=torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self,enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self,*args):
            pass

class ResidualBlock(nn.Module):
    def __init__(self,in_planes,planes,norm_fn='group',stride=1):
        super(ResidualBlock,self).__init__()

        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=3,padding=1,stride=stride)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,padding=1)
        self.relu=nn.ReLU(in_planes=True)

        num_groups=planes//8
        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride==1 and in_planes==planes):
                self.norm3=nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride==1 and in_planes==planes):
                self.norm3=nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride==1 and in_planes==planes):
                self.norm3=nn.InstanceNorm2d(planes)
        else:
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 =nn.Sequential()

        if (stride == 1 and in_planes == planes):
            self.downsample=None
        else:
            self.downsample=nn.Sequential(nn.Conv2d(in_planes,planes,kernel_size=1,stride=stride),self.norm3)

    def forward(self,x):
        y=x
        y=self.conv1(y)
        y=self.norm1(y)
        y=self.relu(y)
        y=self.conv2(y)
        y=self.norm2(y)
        y=self.relu(y)

        if self.downsample is not None:
            x=self.downsample(x)

        return x+y

class BasicEncoder(nn.Module):
    def __init__(self,output_dim=[128],norm_fn='batch',dropout=0.0,downsample=3):
        super(BasicEncoder,self).__init__()
        self.norm_fn=norm_fn
        self.downsample=downsample

        if norm_fn=='group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif norm_fn=='batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif norm_fn=='instance':
            self.norm1 = nn.InstanceNorm2d(64)
        else  :
            self.norm1 = nn.Sequential()

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=1+(downsample>2),padding=3)   #可能进行第三次下采样
        self.relu1=nn.ReLU(inplace=True)
        self.in_planes=64
        self.layer1=self._make_layer(64,stride=1)
        self.layer2=self._make_layer(96,stride=1+(downsample>1)) #进行两次下采样
        self.layer3=self._make_layer(128,stride=1+(downsample>0))
        self.layer4=self._make_layer(128,stride=2)
        self.layer5=self._make_layer(128,stride=2)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1))
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs32 = nn.ModuleList(output_list)

        if dropout>0:
            self.dropout=nn.Dropout2d(p=dropout)
        else:
            self.dropout=None

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,(nn.BatchNorm2d,nn.InstanceNorm2d,nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight,1)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

    def _make_layer(self,dim,stride=1):
        #两个残差层第一个或进行下采样
        layer1=ResidualBlock(self.in_planes,dim,self.norm_fn,stride=stride)
        layer2=ResidualBlock(dim,dim,self.norm_fn,stride=1)
        layers=(layer1,layer2)
        self.in_planes=dim
        return nn.Sequential(*layers)

    def forward(self, x, dual_inp=False, num_layers=3):  #dual_inp代表是否输入的是左右图
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  #得到要求分辨率的特征图
        if dual_inp:
            v=x
            x=x[:(x.shape[0]//2)]


        outputs08 = [f(x) for f in self.outputs08]
        if num_layers == 1:
            return (outputs08, v) if dual_inp else (outputs08,)

        y = self.layer4(x)
        outputs16 = [f(y) for f in self.outputs16]

        if num_layers == 2:
            return (outputs08, outputs16, v) if dual_inp else (outputs08, outputs16)

        z = self.layer5(y)
        outputs32 = [f(z) for f in self.outputs32]

        return (outputs08, outputs16, outputs32, v) if dual_inp else (outputs08, outputs16, outputs32)


class ConvGRU(nn.Module):
    def __init__(self,hidden_dim,input_dim,kernel_size=3):
        super(ConvGRU,self).__init__()
        self.convz=nn.Conv2d(hidden_dim+input_dim,hidden_dim,kernel_size,padding=kernel_size//2)
        self.convr=nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq=nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self,h,cz,cr,cq,*x_list):
        x=torch.cat(x_list,dim=1)
        hx=torch.cat([h,x],dim=1)
        z=torch.sigmoid(self.convz(hx)+cz)
        r=torch.sigmoid(self.convr(hx)+cr)
        q=torch.tanh(self.convq(torch.cat([r*h,x],dim=1))+cq)

        h=(1-z)*h+z*q
        return h


def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)
def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)





class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args

        cor_planes = args.corr_levels * (2*args.corr_radius + 1)

        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class BasicMultiUpdateBlock(nn.Module):
    def __init__(self,args,hidden_dims=[]):
        super().__init__()
        self.args=args
        self.encoder=BasicMotionEncoder(args)
        encoder_output_dim=128
        self.gru08 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru16 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru32 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.flow_head = FlowHead(hidden_dims[2], hidden_dim=256, output_dim=2)
        factor = 2**self.args.n_downsample

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor**2)*9, 1, padding=0))
    def forward(self, net, inp, corr=None, flow=None, iter08=True, iter16=True, iter32=True, update=True):

        if iter32:
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if iter16:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]))
        if iter08:
            motion_features = self.encoder(flow, corr)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_flow = self.flow_head(net[0])

        # scale mask to balence gradients
        mask = .25 * self.mask(net[0])
        return net, mask, delta_flow

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    assert torch.unique(ygrid).numel() == 1 and H == 1 # This is a stereo problem

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

class CorrBlock1D:
    def __init__(self,fmap1,fmap2,num_levels=4,radius=4):
        self.num_levels=num_levels
        self.radius=radius
        self.corr_pyramid=[]
        corr=CorrBlock1D.corr(fmap1,fmap2)
        batch,h1,w1,dim,w2=corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, 1, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels):
            corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])
            self.corr_pyramid.append(corr)                  #得到不同分辨率的相关金字塔

    def __call__(self, coords):  #coords:N*2*H1*W1
        r=self.radius
        coords=coords[:,:1].permute(0,2,3,1)       #N*H1*W1*1
        batch,h1,w1,_=coords.shape

        out_pyramid=[]
        for i in range(self.num_levels):
            corr=self.corr_pyramid[i]
            dx=torch.linspace(-r,r,2*r+1)
            dx=dx.view(1,1,2*r+1,1).to(coords.device)
            x0=dx+coords.reshape(batch*h1*w1,1,1,1)/2**i
            y0=torch.zeros_like(x0)     #x0、y0:（B*H1*W1）*1*2r+1*1

            coords_lvl = torch.cat([x0, y0], dim=-1)       #coords_lvl(B*H1*W1)*1*2r+1*2,coor:(B*H1*W1)*1*1*W2
            corr = bilinear_sampler(corr, coords_lvl)      #返回(B*H1*W1)*1*1*2r+1
            corr = corr.view(batch, h1, w1, -1)            #整合成B*H1*W1*2r+1
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()     #返回B*C*H1*W1,其中C为num_level*(2*r+1)

    @staticmethod
    def corr(fmap1,fmap2):
        B,D,H,W1=fmap1.shape
        _,_,_,W2=fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr=torch.einsum('aijk,aijh->ajkh',fmap1,fmap2)
        corr=corr.reshape(B,H,W1,1,W2).continguous()
        return corr/torch.sqrt(torch.tensor(D).float)


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


class RAFTStereo(nn.Module):
    def __init__(self,args):
        super(RAFTStereo,self).__init__()
        self.args=args
        context_dims=args.hidden_dims #隐藏层维数和文本信息维数
        self.cnet=BasicEncoder(output_dim=[args.hidden_dims,context_dims],norm_fn="batch",downsample=args.n_downsample)  #特征提取，输出不同分辨率的context和左右特征图
        self.update_block=BasicMultiUpdateBlock(self.args.hidden_dims==args.hidden_dims)
        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], args.hidden_dims[i] * 3, 3, padding=3 // 2) for i in
             range(self.args.n_gru_layers)])          #把context送进GRU,为什么输出是隐藏层的通道数的3倍？
        self.conv2=nn.Sequential(ResidualBlock(128,128,'instance',stride=1),nn.Conv2d(128,256,3,padding=1))

    def initialize_flow(self,img):
        N,_,H,W=img.shape
        coords0=coords_grid(N,H,W).to(img.device)
        coords1=coords_grid(N,H,W).to(img.device)  #coords大小为N*2*H*W
        return coords0,coords1


    def forward(self,image1,image2,iters=12,flow_init=None,test_mode=False):
        image1=(2*(image1/255.0)-1.0).contiguous()
        image2=(2*(image2/255.0)-1.0).contiguous()

        with autocast(enabled=self.args.mixed_precision):
            *cnet_list,x=self.cnet(torch.cat(image1,image2,dim=0),dual_inp=True,num_layers=self.args.n_gru_layers)
            fmap1,fmap2=self.conv2(x).split(dim=0,split_size=x.shape[0]//2)

            net_list=[torch.tanh(x[0]) for x in cnet_list]
            inp_list=[torch.relu(x[1]) for x in cnet_list]
            #inp_list是给GRU隐藏状态的输入
            inp_list=[list(conv(i).split(dim=1,split_size=conv.out_chasnnels//3)) for i,conv in zip(inp_list,self.context_zqr_convs)]
        corr_block=CorrBlock1D
        corr_fn=corr_block(fmap1,fmap2,radis=self.args.corr_radius,num_levels=self.args.corr_levels)
        coords0,coords1=self.initialize_flow(net_list[0])  #N*2*H1*W1

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions=[]
        for itr in range(iters):
            coords1=coords1.detach()
            corr=corr_fn(coords1)
            flow=coords1-coords0
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)










