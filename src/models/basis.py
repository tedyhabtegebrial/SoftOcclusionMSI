import torch

class BasisFunction(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_spheres_or_occlusions = configs.num_layers
        if configs.num_basis>1:
            self.layers = self._create_basis_layers()
        self.register_buffer('ones', torch.ones(1, self.num_spheres_or_occlusions, configs.height, configs.width))
    def _create_basis_layers(self):
        configs = self.configs
        num_inputs = 3
        num_l = configs.basis_layers
        num_b = (configs.num_basis-1)*self.num_spheres_or_occlusions
        input_sizes = [int(i.item()) for i in torch.linspace(num_inputs, num_b, num_l)]
        output_sizes = [*input_sizes[1:]] + [num_b]
        layers = []
        for i,o in zip(input_sizes, output_sizes):
            layers.append(torch.nn.Conv2d(i, o, kernel_size=1, stride=1, padding=0))
        return torch.nn.ModuleList(layers)
    
    def forward(self, input_vecs):
        if self.configs.num_basis==1:
            b = input_vecs.shape[0]
            c, h, w = self.ones.shape[1:]
            ones = self.ones.expand(b, c, h, w)
            return ones
        x = input_vecs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if not i==(len(self.layers)-1):
                x = torch.relu(x)
        x = torch.tanh(x)
        b = x.shape[0]
        c, h, w = self.ones.shape[1:]
        ones = self.ones.expand(b, c, h, w)
        x = torch.cat([x, ones], dim=1)
        return x

if __name__== '__main__':
    class Obj:
        def __init__(self, b, l, f):
            self.basis_layers = l
            self.num_basis = b
            self.feats_per_layer = f
    obj = Obj(8, 4, 3)
    net = BasisFunction(obj)
    data = torch.rand(1, 3, 32, 31)
    res = net(data)
    print(res.shape)
    print(res.min(), res.max())
