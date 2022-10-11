import math
import torch
import copy
import functools

import torch.nn.functional as F
from torch.distributions.uniform import Uniform


class Utils(object):
    def __init__(self, configs):
        super().__init__()
        dataset = configs.dataset
        # batch_size = configs.batch_size
        height = configs.height
        width = configs.width
        self.height, self.width = height, width
        # self.batch_size = batch_size
        self.get_xy_coords_cache = {}
        self.erp_2_sphere_cache = {}
        self.dataset = dataset
        self.unit_rays_cache = None
        self.theta_dist = Uniform(
            0*math.pi*torch.ones(1), 2*math.pi*torch.ones(1))
        self.phi_dist = Uniform(-0.1*math.pi*torch.ones(1), -
                                0.1*math.pi*torch.ones(1))

    def get_xy_coords(self, device='cpu', no_cache=False):
        """Short summary.
        :return: Description of returned object.
        :rtype: torch.Tensor of shape [B, H, W, 2]
        """
        if len(self.get_xy_coords_cache.keys()) > 0 and (not no_cache):
            return copy.deepcopy(self.get_xy_coords_cache['any'])
        height, width = self.height, self.width
        batch_size = 1
        x_locs = torch.linspace(0, width-1, width).view(1, width, 1)
        y_locs = torch.linspace(0, height-1, height).view(height, 1, 1)
        x_locs, y_locs = map(lambda x: x.to(device), [x_locs, y_locs])
        x_locs, y_locs = map(lambda x: x.expand(
            height, width, 1), [x_locs, y_locs])
        xy_locs = torch.cat([x_locs, y_locs], dim=2)
        xy_locs = xy_locs.unsqueeze(0).expand(batch_size, height, width, 2)
        self.get_xy_coords_cache['any'] = xy_locs
        return xy_locs

    def equi_2_spherical(self, equi_coords, radius=1, no_cache=False):
        """Short summary
        :param: torch.Tensor: equi_coords of shape [B, H, W, 2]
        :radius: torch.Tensor: optional radius values, of shape [B] or [1]
        :return:
        :equi_coords: spherical cooridnates of shape [B, H, W, 3], the last cooridnates contains azimuth, elevation and radius
        """
        # if self.dataset in self.erp_2_sphere_cache.keys() and (not no_cache):
        #     return self.erp_2_sphere_cache[self.dataset].clone()
        height, width = self.height, self.width
        input_shape = equi_coords.shape
        assert input_shape[-1] == 2, 'last coordinate should be 2'
        if self.dataset in ['replica', 'dense_replica']:
            x_locs, y_locs = torch.split(
                tensor=equi_coords, dim=-1, split_size_or_sections=1)
            # x_locs = x_locs.clamp(min=0, max=width-1)
            # y_locs = y_locs.clamp(min=0, max=height-1)
            theta = (2*math.pi / (width-1)) * x_locs - math.pi
            # print('real_theta:', theta.min()/math.pi, theta.max()/math.pi)
            # print('expected:', -1, 1)
            # 2*(math.pi/width)*x_locs - 1.5*math.pi
            phi = (math.pi/(height-1))*(y_locs)
            spherical_coords = torch.cat(
                [theta, phi, torch.ones_like(theta).mul(radius)], dim=-1)
            self.erp_2_sphere_cache[self.dataset] = spherical_coords.clone()
        elif self.dataset in ['d3dkit', 'coffee', 'coffee_1d', 'coffee_2d', 'coffee_12v', 'coffee_r2', 'coffee_r3', 'coffee_r1', 'coffee3x3']:
            x_locs, y_locs = torch.split(
                tensor=equi_coords, dim=-1, split_size_or_sections=1)
            x_locs = x_locs.clamp(min=0, max=width-1)
            y_locs = y_locs.clamp(min=0, max=height-1)
            theta = (-2*math.pi / (width-1)) * x_locs + 2*math.pi
            phi = (math.pi/(height-1))*(y_locs)
            spherical_coords = torch.cat(
                [theta, phi, torch.ones_like(theta).mul(radius)], dim=-1)
            # self.erp_2_sphere_cache[self.dataset] = spherical_coords.clone()
        elif self.dataset == 'residential':
            x_locs, y_locs = torch.split(
                tensor=equi_coords, dim=-1, split_size_or_sections=1)
            x_locs = x_locs.clamp(min=0, max=width-1)
            y_locs = y_locs.clamp(min=0, max=height-1)
            theta = math.pi*(2*x_locs/(width-1) - 1.5)
            phi = math.pi*(0.5-y_locs/(height-1))
            spherical_coords = torch.cat(
                [theta, phi, torch.ones_like(theta).mul(radius)], dim=-1)
            self.erp_2_sphere_cache[self.dataset] = spherical_coords.clone()
        else:
            raise NotImplementedError(
                f'dataset={self.dataset} is not implemented yet')
        return spherical_coords

    def spherical_2_cartesian(self, spherical_coords):# checked
        '''
        No caching for this function
        :param: torch.Tensor: spherical_coords should be of shape [B, H, W, 3] or [B, H, W, 2] (when radius is not passed, it will be assumed to be 1)
        :return:
        :xyz_locs: torch.Tensor: cartesian coordiantes with shape [B, H, W, 3]
        '''
        input_shape = spherical_coords.shape
        assert input_shape[-1] in [2,
                                   3], 'last dimension of input should be 3 or 2'
        coordinate_split = torch.split(
            spherical_coords, split_size_or_sections=1, dim=-1)
        theta, phi = coordinate_split[:2]
        if input_shape[-1] == 3:
            rad = coordinate_split[2]
        else:
            rad = torch.ones_like(theta).to(theta.device)
        if self.dataset in ['replica', 'd3dkit', 'coffee', 'coffee_1d', 'coffee_2d', 'coffee_12v', 'coffee_r2', 'coffee_r3', 'coffee_r1', 'coffee3x3', 'dense_replica']:
            x_locs = rad * torch.sin(phi) * torch.cos(theta)
            y_locs = rad * torch.sin(phi) * torch.sin(theta)
            z_locs = rad * torch.cos(phi)
        elif self.dataset == 'residential':
            # theta
            x_locs = rad * torch.cos(theta) * torch.cos(phi)
            z_locs = rad * torch.sin(theta) * torch.cos(phi)
            y_locs = rad * torch.sin(phi)
        xyz_locs = torch.cat([x_locs, y_locs, z_locs], dim=-1)
        return xyz_locs

    def cartesian_2_spherical(self, input_points, normalized=False):
        '''conversion from cartesian to sphericall coordinates
        :param input_points: tensor of shape [B, ..., 3] or [B, ..., 3, 1]
        :param normalized: boolean is True the input_points will be treated as unit vectors
        '''
        last_coord_one = False
        if input_points.shape[-1] == 1:
            input_points = input_points.squeeze(-1)
            last_coord_one = True
        if not normalized:
            input_points = self.normalize_3d_vectors(input_points)
        x_c, y_c, z_c = torch.split(
            input_points, split_size_or_sections=1, dim=-1)
        r = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
        if self.dataset in ['replica', 'dense_replica']:
            theta = torch.atan2(y_c, x_c)
            # print('theta range real    : ', theta.min().item()/math.pi, theta.max().item()/math.pi)
            phi = torch.acos(z_c/r)
            mask1 = theta.gt(math.pi)
            # torch.logical_and(
            # , theta.le(-1*math.pi))
            theta[mask1] = theta[mask1] - 2*math.pi
            mask2 = theta.lt(-1*math.pi)
            # torch.logical_and(
            # , theta.le(-1*math.pi))
            theta[mask2] = theta[mask2] + 2*math.pi
            
        elif self.dataset in ['d3dkit', 'coffee', 'coffee_1d', 'coffee_2d', 'coffee_12v', 'coffee_r2', 'coffee_r3', 'coffee_r1', 'coffee3x3']:
            theta = torch.atan2(y_c, x_c)
            phi = torch.acos(z_c/r)
            mask1 = theta.lt(0)
            theta[mask1] = theta[mask1] + 2*math.pi
            # mask2 = theta.lt(0)
            # theta[mask2] = theta[mask2] + 2*math.pi
            # print('phi: ', phi.min().item(), phi.max().item())
            # print('theta: ', theta.min().item(), theta.max().item())
        elif self.dataset == 'residential':
            theta = -torch.atan2(-z_c, x_c)
            phi = torch.asin(y_c/r)
            mask = torch.logical_and(
                theta.gt(math.pi*0.5), theta.le(2*math.pi))
            theta[mask] = theta[mask] - 2*math.pi
        spherical_coords = torch.cat(
            [theta, phi, torch.ones_like(theta)], dim=-1)
        return spherical_coords

    def spherical_2_equi(self, spherical_coords, height=None, width=None):
        """spherical coordinates to equirectangular coordinates
        :param spherical_coords: tensor of shape [B, ..., 3], [B, ..., 3, 1], [B, ..., 2] or [B, ..., 2, 1], 
        :param height: image height, optional when not given self.height will be used
        :param width: image width, optional when not given self.width will be used
        """
        height = height if not height is None else self.height
        width = width if not width is None else self.width
        last_coord_one = False
        if spherical_coords.shape[-1] == 1:
            spherical_coords = spherical_coords.squeeze(-1)
            last_coord_one = True
        spherical_coords = torch.split(
            spherical_coords, split_size_or_sections=1, dim=-1)
        theta, phi = spherical_coords[0], spherical_coords[1]
        # print('theta, phi --- min:', theta.min(), phi.min())
        # print('theta, phi --- max:', theta.max(), phi.max())
        if self.dataset in ['replica', 'dense_replica']:
            x_locs = ((width-1)/(2.0*math.pi)) * (theta + math.pi)
            y_locs = (height-1)/math.pi * phi
        elif self.dataset in ['d3dkit', 'coffee', 'coffee_1d', 'coffee_2d', 
                                'coffee_12v', 'coffee_r2', 'coffee_r3', 'coffee_r1', 'coffee3x3']:
            x_locs = (width-1) * (1 - theta/(2.0*math.pi))
            y_locs = phi*(height-1)/math.pi
            # print('locs')
            # print('x_locs', x_locs.data.min())
            # print('x_locs', x_locs.data.max())
            # print('y_locs', y_locs.data.min())
            # print('y_locs', y_locs.data.max())
        elif self.dataset == 'residential':
            # theta = torch.clamp(theta, min=-1.5*math.pi, max=0.5*math.pi)
            # phi = torch.clamp(phi, min=-math.pi/2, max=math.pi/2)
            # x_locs = ((width-1)/(2.0*math.pi))*theta + (3/4.0)*(width-1)
            # y_locs = (height-1)*(0.5 - phi/math.pi)
            #
            # theta = torch.clamp(theta, min=-1.5*math.pi, max=0.5*math.pi)
            # phi = torch.clamp(phi, min=-math.pi/2, max=math.pi/2)
            x_locs = ((1/(2.0*math.pi))*theta + (3/4.0))*(width-1)
            y_locs = (0.5 - phi/math.pi)*(height-1)
            #
            # print('locs')
            # print('x_locs', x_locs.data.min())
            # print('x_locs', x_locs.data.max())
            # print('y_locs', y_locs.data.min())
            # print('y_locs', y_locs.data.max())

        else:
            raise NotImplementedError(
                f'Dataset {self.dataset} is not implemented')
        xy_locs = torch.cat([x_locs, y_locs], dim=-1)
        if last_coord_one:
            xy_locs = xy_locs.unsqueeze(-1)
        return xy_locs

    def scale_vectors(self, input_vectors, scale, normalize=False):
        '''input vectors and scale value should have dims that allow element-wise multiply
        :param input_vectors: input vectors in 3D, of shape [B, H, W, 3] or [B, D, H, W, 3]
        :type input_vectors: torch.Tensor
        :param scale: scaling values of shape [B, H, W, 1], [B, D, H, W, 1] or [B], or [1] (a float value)
        :type scale: torch.Tensor, float
        '''
        vec_shape = input_vectors.shape
        unit_vectors = input_vectors if not normalize else self.normalize_3d_vectors(
            input_vectors)
        scaled_vectors = scale*unit_vectors
        return scaled_vectors

    def transform_3d_points(self, input_points, rot_mats, t_vecs):
        """
        :param input_points: points in 3D os shape [B, H, W, 3] or [B, D, H, W, 3]
        :type input_poitns: torch.Tensor
        :param rot_mats: tensor of shape [B, 3, 3]
        :type rot_mats: torch.Tensor
        :param t_vecs: tensor of shape [B, 3, 1]
        :type t_vecs: torch.Tensor
        """
        input_shape = input_points.shape
        assert len(
            rot_mats.shape) == 3, 'Rotation matrix should be of shape [B, 3, 3]'
        assert len(
            t_vecs.shape) == 3, 'Translation matrix should be of shape [B, 3, 1]'
        assert rot_mats.shape[0] == input_shape[
            0], f'Rotation matrix and input_shape should have same batch size got {rot_mats.shape}_{input_shape}'
        input_points = input_points.unsqueeze(-1)
        # print(input_points.shape, rot_mats.shape, t_vecs.shape)
        # exit()
        for _ in range(len(input_points.shape)-len(t_vecs.shape)):
            t_vecs = t_vecs.unsqueeze(1)
            rot_mats = rot_mats.unsqueeze(1)
        transformed_points = torch.matmul(rot_mats, input_points) + t_vecs
        transformed_points = transformed_points.squeeze(-1)
        return transformed_points

    def normalize_3d_vectors(self, input_points, p=2, eps=1e-12):
        '''normalises input 3d points along the last dimension
        :param input_points: 3D points of shape [B, ..., 3]
        :param p: norm power
        :param eps: epsilone to avoid division by 0
        '''
        input_shape = input_points.shape
        last_coord_one = False
        if input_shape[-1] == 1:
            last_coord_one = True
            input_points = input_points.squeeze(-1)
        p_norm = torch.norm(input_points, p=p, dim=-1,
                            keepdim=True).clamp(min=eps)
        normalized_points = input_points / p_norm
        if last_coord_one:
            normalized_points = normalized_points.unsqueeze(-1)
        return normalized_points

    def safe_ray_sphere_intersection(self, ray_origin, direction, sphere_radii):
        """ this is same as ray_sphere_intersection, but we know that the sphere and ray intersect 
            at exactly one point
            print(ray_origin.shape, direction.shape, sphere_radii.shape)
            torch.Size([2, 256, 512, 3]) torch.Size([2, 256, 512, 3]) torch.Size([2, 32])
        """
        # print(ray_origin.shape, direction.shape, sphere_radii.shape)
        # exit()
        assert ray_origin.shape == direction.shape, 'shapes of origin and direction should'
        assert len(sphere_radii.shape) in [
            2, 4], 'sphere radii should be 2 or 4 D'
        input_shape = ray_origin.shape
        b_size, height, width, _ = ray_origin.shape
        num_spheres = sphere_radii.shape[-1]
        if len(sphere_radii.shape) == 2:
            # change sphere_radii  to shape [B, D, 1, 1]
            sphere_radii = sphere_radii.unsqueeze(
                -1).unsqueeze(-1).unsqueeze(-1)
        # change ray and direction to shape [B, 1, H, W, 3]
        # .expand(b_size, num_spheres, height, width, 3)
        ray_origin = ray_origin.unsqueeze(1)
        # .expand(b_size, num_spheres, height, width, 3)
        direction = direction.unsqueeze(1)
        # solve the quadratic equation of ray sphere intersection
        # torch.sum(torch.mul(direction, direction), dim=-1, keepdim=True)
        a = 1
        b = 2*torch.mul(direction, ray_origin).sum(dim=-1, keepdim=True)
        # .sum(dim=-1, keepdim=True)
        r_square = torch.mul(sphere_radii, sphere_radii)
        r_square = r_square.expand(b_size, num_spheres, height, width, 1)
        c = torch.mul(ray_origin, ray_origin).sum(
            dim=-1, keepdim=True) - r_square
        discriminant_ = torch.mul(b, b).sum(dim=-1, keepdim=True) - 4*a*c
        # make sure disciminant is positive
        
        # print('sphere_radii', sphere_radii)
        # print('discriminant_:', discriminant_.min(), discriminant_.max())
        # exit()
        discriminant = discriminant_  # .clamp(min=0.0)
        sqrt_discriminant = torch.sqrt(discriminant)
        denominator = (2.0*a)  # .clamp(min=1e-08)
        # t_values = (-b - sqrt_discriminant) / denominator
        t_values = (-b + sqrt_discriminant) / denominator
        intersection = ray_origin + torch.mul(t_values, direction)
        return intersection  # .view(b_size, num_spheres, height, width, 3)

    def ray_sphere_intersection(self, ray_origin, direction, sphere_radii):
        """
        :param torch.FloatTensor ray_origin: 3D points for the origin of each ray
        ray_origin shape [B, H, W, 3]
        :param torch.FloatTensor direction: 3D points where each ray ends
        direction shape [B, H, W, 3]
        :param torch.FloatTensor sphere_radii: radii of each shere
        [B, 1] or [B, D], [B, D, H, W], [B, 1, H, W]
        :return: Description of returned object.
        :rtype: torch.FloatTensor intersection: points of intersection
        intersection shape [B, D, H, W, 3]
        :rtype: torch.LongTensor mask: validity of each itnersection point
        mask shape [B, D, H, W, 1]
        """
        assert ray_origin.shape == direction.shape, 'shapes of origin and direction should'
        assert len(sphere_radii.shape) in [
            2, 4], 'sphere radii should be 2 or 4 D'
        input_shape = ray_origin.shape
        b_size, height, width, _ = ray_origin.shape
        num_spheres = sphere_radii.shape[-1]
        if len(sphere_radii.shape) == 2:
            # change sphere_radii  to shape [B, D, 1, 1]
            sphere_radii = sphere_radii.unsqueeze(-1).unsqueeze(-1)
        # change ray and direction to shape [B, 1, H, W, 3]
        # .expand(b_size, num_spheres, height, width, 3)
        ray_origin = ray_origin.unsqueeze(1)
        # .expand(b_size, num_spheres, height, width, 3)
        direction = direction.unsqueeze(1)
        # solve the quadratic equation of ray sphere intersection
        a = torch.sum(torch.mul(direction, direction), dim=-1, keepdim=True)
        b = 2*torch.mul(direction, ray_origin).sum(dim=-1, keepdim=True)
        r_square = torch.mul(sphere_radii, sphere_radii).sum(
            dim=-1, keepdim=True)
        c = torch.mul(ray_origin, ray_origin).sum(
            dim=-1, keepdim=True) - r_square
        discriminant = torch.mul(b, b).sum(dim=-1, keepdim=True) - 4*a*c
        mask = discriminant.lt(0)
        not_mask = torch.logical_not(mask)
        discriminant[mask] = 0.0
        sqrt_discriminant = torch.sqrt(discriminant)
        denominator = (2.0*a).clamp(min=1e-07)
        sol_1 = (-b + sqrt_discriminant) / denominator
        sol_2 = (-b - sqrt_discriminant) / denominator
        sol_1_mask = sol_1 < sol_2
        sol_2_mask = torch.logical_not(sol_1_mask)
        solution = torch.zeros_like(sqrt_discriminant)
        solution[sol_1_mask] = sol_1[sol_1_mask]
        solution[sol_2_mask] = sol_2[sol_2_mask]
        t_values = torch.zeros_like(solution)
        t_values[not_mask] = solution[not_mask]
        # now we need intesection points from t, ray_origin and direction
        intersection = ray_origin + torch.mul(t_values, direction)
        # ignore any negative value of t
        neg_t = t_values.lt(0)
        not_mask[neg_t] = False
        return intersection.view(b_size, num_spheres, height, width, 3), not_mask

    def get_rel_pose(self, src_pose, trg_pose, camera_2_world=True):
        """returns ralative camera pose from src to trg pose
        :param src_pose: a tuple/list of source camera rotation and translation
        :param trg_pose: a tuple/list of target camera rotation and translation
        :param camera_2_world: boolean, if True rotation matrices are interpreted as from camera->world
        """
        # rotation is defined from camera to world
        # camera to world
        if camera_2_world:
            r_src, t_src = src_pose
            r_trg, t_trg = trg_pose
            r_rel = torch.bmm(torch.inverse(r_trg), r_src)
            t_rel = torch.bmm(torch.inverse(r_trg), (t_src - t_trg))
        else:
            # world to camera
            r_src, t_src = src_pose
            r_trg, t_trg = trg_pose
            r_rel = torch.bmm(r_trg, r_src.permute(0, 2, 1))
            t_rel = torch.bmm(r_rel, t_src) - t_trg
        return r_rel, t_rel

    def normalize_grid(self, grid):
        has_D = len(grid.shape) == 5
        if has_D:
            b_size, num_d_chans = grid.shape[0], grid.shape[1]
            grid = grid.view(
                b_size*grid.shape[1], grid.shape[2], grid.shape[3], grid.shape[4])
        b = grid.shape[0]
        h, w = self.height, self.width
        x_locs, y_locs = torch.split(grid, split_size_or_sections=1, dim=-1)
        x_min = x_locs.min().item()
        x_max = x_locs.max().item()
        y_min = y_locs.min().item()
        y_max = y_locs.max().item()

        assert x_min>=0, f'xmin found to be{x_min}'
        assert x_max<(w), f'xmin found to be{x_max}'
        assert y_min>=0, f'xmin found to be{y_min}'
        assert y_max<(h), f'xmin found to be{y_max}'

        x_locs = torch.clamp(x_locs , min=0.0, max=w-1)
        y_locs = torch.clamp(y_locs, min=0.0, max=h-1)
        x_half = (w-1)/2.0
        y_half = (h-1)/2.0
        u_locs =  (x_locs - x_half) / x_half
        v_locs =  (y_locs - y_half) / y_half
        # v_locs = -1 + 2*y_locs/h
        norm_grid = torch.cat([u_locs, v_locs], dim=-1)
        if has_D:
            norm_grid = norm_grid.view(
                b_size, num_d_chans, grid.shape[1], grid.shape[2], grid.shape[3])
        return norm_grid

    def sample_equi(self, input, grid, mode='bilinear', normalize_grid=True):
        """Samples input equi-rectangula image using a sampling grid
        :param input: tensor of shape [B, F, H, W] or [B, D, F, H, W] # D is when we process sphere sweep volumes
        :param grid: tensor of shape [B, H, W, 2] or [B, D, H, W, 2]
        :param normalize_grid: if true grid locs will be converted to range [-1, 1]
        """
        assert grid.shape[-1] == 2, 'grid last dim should be 2'
        h, w = input.shape[-2:]
        b_size = input.shape[0]
        assert h == self.height, 'input/grid height should be same as self.height'
        assert w == self.width, 'input/grid width should be same as self.width'
        has_D = len(input.shape) == 5
        if has_D:
            # assert len(grid.shape)==5, 'input grid should be 5D too'
            num_d_chans = input.shape[1]
            input = input.view(
                b_size*input.shape[1], input.shape[2], input.shape[3], input.shape[4])
            grid = grid.view(
                b_size*grid.shape[1], grid.shape[2], grid.shape[3], grid.shape[4])
        norm_grid = self.normalize_grid(grid) if normalize_grid else grid
        # (-1 and 1) refer to the corner points the pixels at the image borders | not their center points
        sampled = F.grid_sample(input=input, grid=norm_grid, mode=mode,
                                align_corners=True, padding_mode='zeros')
        if has_D:
            sampled = sampled.view(
                b_size, num_d_chans, sampled.shape[1], sampled.shape[2], sampled.shape[3])
        return sampled

    def unproject_perspective(self, points_2d, k_matrix, normalize=False):
        '''
        :param points_2d should be of shape [B, H, W, 2]
        :type poitns_2d: torch.Tensor
        :param: k_matrix: intrisic matrix with shape [B, 3, 3]
        :type k_matrix: torch.Tensor
        :param: normalize:if True the poitns will be normalized before returning,
                                    this is useful when later the values are multiplied by depth map that contains ray length
        :type normalize: bool
        :rtype rays: rays in 3d in the camera coordinate system with shape
        :type rays: torch.Tensor
        '''
        b, h, w, _ = points_2d.shape
        device = points_2d.device
        points_2d_h = torch.cat([points_2d, torch.ones(
            b, h, w, 1).to(device)], dim=3).unsqueeze(4)
        k_inv = torch.inverse(k_matrix)
        k_inv = k_inv.view(b, 1, 1, 3, 3)
        rays = torch.matmul(k_inv, points_2d_h).squeeze(4)
        if normalize:
            rays_norm = torch.norm(
                rays, p=2, dim=3, keepdim=True).clamp(min=1e-08)
            rays = rays/rays_norm
        return rays

    def get_random_theta_and_phi(self, device='cpu'):
        return self.theta_dist.sample().to(device), self.phi_dist.sample().to(device)

    def rot_around(self, angle, axis='x'):
        device = angle.device
        b_size = angle.shape[0]
        ones = torch.ones(b_size).to(device)
        rot_mat = torch.zeros(b_size, 3, 3).to(device)
        if axis == 'x':
            rot_mat[:, 0, 0] = ones
            rot_mat[:, 1, 1] = torch.cos(angle)
            rot_mat[:, 1, 2] = -1*torch.sin(angle)
            rot_mat[:, 2, 1] = torch.sin(angle)
            rot_mat[:, 2, 2] = torch.cos(angle)
        elif axis == 'y':
            rot_mat[:, 1, 1] = ones
            rot_mat[:, 0, 0] = torch.cos(angle)
            rot_mat[:, 0, 2] = torch.sin(angle)
            rot_mat[:, 2, 0] = -1*torch.sin(angle)
            rot_mat[:, 2, 2] = torch.cos(angle)
        elif axis == 'z':
            rot_mat[:, 2, 2] = ones
            rot_mat[:, 0, 0] = torch.cos(angle)
            rot_mat[:, 0, 1] = -1*torch.sin(angle)
            rot_mat[:, 1, 0] = torch.sin(angle)
            rot_mat[:, 1, 1] = torch.cos(angle)
        return rot_mat

    def get_rotation_matrices(self, theta, phi):
        '''
        Gets tensors of thetas and phis as input and returns a rotation matrix
        that maps a unit vector pointing in the direction of | azimuth=0, elevation=0 |
        to a vector pointing at | azimuth=theta, elevation=phi |
        '''
        theta, phi = theta.view(-1), phi.view(-1)
        if self.dataset == 'residential':
            r_z = self.rot_around(phi, axis='z')
            r_y = self.rot_around(math.pi*2 - theta, axis='y')
            rot_mat = torch.bmm(r_y, r_z)
        elif self.dataset in ['replica', 'dense_replica']:
            r_y = self.rot_around(phi, axis='y')
            r_z = self.rot_around(theta, axis='z')
            rot_mat = torch.bmm(r_z, r_y)
        else:
            raise NotImplementedError
        return rot_mat
        # Step 1 get src vectors
        # phi_src = torch.zeros_like(phi)
        # theta_src = torch.zeros_like(theta)
        # radius = torch.ones_like(theta)
        # spherical_vecs_src = torch.stack([theta_src, phi_src, radius], dim=-1)
        # spherical_vecs_trg = torch.stack([theta, phi, radius], dim=-1)
        # cartesian_vectors_src = self.spherical_2_cartesian(spherical_vecs_src).unsqueeze(-1)
        # cartesian_vectors_trg = self.spherical_2_cartesian(
        #     spherical_vecs_trg).unsqueeze(-1)
        # # Rotation matrix: that maps src to trg
        # h_mat = torch.bmm(cartesian_vectors_src, cartesian_vectors_trg.permute(0, 2, 1))
        # U,_,V = torch.svd(h_mat)
        # rot_mat = torch.bmm(V, U.permute(0, 2, 1))
        # return rot_mat, cartesian_vectors_trg, cartesian_vectors_src
        # return rot_mat

    def rotate_features(self, features, rot_mats):
        b, c, h, w = features.shape
        # theta, phi = theta_phi
        # phi = phi.view(b, 1, 1, 1)
        # theta = theta.view(b, 1, 1, 1)
        # erp_locs = self.get_xy_coords(device=features.device)
        # spherical_coords = self.equi_2_spherical(erp_locs)
        # theta_can, phi_can, rad_can = torch.split(spherical_coords, dim=-1, split_size_or_sections=1)
        # theta_can_shifted, phi_can_shifted = theta_can + theta, phi_can + phi
        # theta_can_shifted, phi_can_shifted = theta_can_shifted%(2*math.pi), phi_can_shifted%(2*math.pi)
        # if self.dataset=='residential':

        #     mask_1 = theta_can_shifted.ge(math.pi*0.5)
        #     theta_can_shifted[mask_1] = theta_can_shifted[mask_1] - 2*math.pi
        #     mask_2 = theta_can_shifted.lt(-1*math.pi*1.5)
        #     theta_can_shifted[mask_2] = theta_can_shifted[mask_2] + 2*math.pi
        #     # phi
        #     mask_3 = phi_can_shifted.lt(-1*math.pi*0.5)
        #     phi_can_shifted[mask_3] = phi_can_shifted[mask_3] + 2*math.pi
        #     mask_4 = phi_can_shifted.ge(math.pi*0.5)
        #     phi_can_shifted[mask_4] = phi_can_shifted[mask_4] - 2*math.pi

        # elif self.dataset=='replica':
        #     mask_1 = theta_can_shifted.ge(math.pi)
        #     theta_can_shifted[mask_1] = theta_can_shifted[mask_1] - 2*math.pi
        #     mask_2 = theta_can_shifted.lt(-1*math.pi*1)
        #     theta_can_shifted[mask_2] = theta_can_shifted[mask_2] + 2*math.pi
        #     # phi
        #     mask_3 = phi_can_shifted.lt(0)
        #     phi_can_shifted[mask_3] = phi_can_shifted[mask_3] + 2*math.pi
        #     mask_4 = phi_can_shifted.ge(math.pi*0.5)
        #     phi_can_shifted[mask_4] = phi_can_shifted[mask_4] - 2*math.pi
        # else:
        #     raise NotImplementedError
        # spherical_coords_shifted = torch.cat([theta_can_shifted, phi_can_shifted, rad_can], dim=3)
        # erp_locs_shifted = self.spherical_2_equi(spherical_coords_shifted)
        # sampled_features = self.sample_equi(
        #     input=features, grid=erp_locs_shifted, normalize_grid=True)
        # # print(theta_can.shape)
        # # exit()
        # # x_locs, y_locs = torch.split(tensor=equi_coords, dim=-1, split_size_or_sections=1)
        unit_rays = self._get_unit_rays_on_sphere(features.device)
        rot_unit_rays = torch.matmul(
            rot_mats.view(b, 1, 1, 3, 3), unit_rays.view(b, h, w, 3, 1)).squeeze(-1)
        rot_spherical = self.cartesian_2_spherical(rot_unit_rays)
        rot_equi = self.spherical_2_equi(rot_spherical)
        sampled_features = self.sample_equi(
            input=features, grid=rot_equi, normalize_grid=True)
        return sampled_features

    def spherical_weighting(self, device='cpu'):
        raise NotImplementedError

    def perspective_cutout(self, input_imgs, h, w, elevation, azimuth, device='cpu', hfov=math.radians(120), vfov=math.radians(90)):
        '''
        elevation: rotation upwards from the horizontal plane
        azimuth: rotation to the right direction from the forward axis
        '''
        assert elevation >= (-0.5*math.pi) and elevation <= (0.5 *
                                                             math.pi), 'wrong elevation range'
        assert azimuth >= (-2*math.pi) and azimuth <= (2 *
                                                       math.pi), 'wrong azimuth range'
        b = input_imgs.shape[0]
        fx = self.width / (2*math.pi)
        fy = self.height / (2*math.pi)
        h = int(2*fy*math.tan(0.5*vfov))
        w = int(2*fx*math.tan(0.5*hfov))
        x_locs = torch.linspace(0, w-1, w).view(1, 1, w, 1).to(device)
        y_locs = torch.linspace(0, h-1, h).view(1, h, 1, 1).to(device)
        x_locs = x_locs.expand(b, h, w, 1)
        y_locs = y_locs.expand(b, h, w, 1)
        hom_2d = torch.cat(
            [x_locs, y_locs, torch.ones_like(x_locs)], dim=3).unsqueeze(-1)

        k_matrix = torch.FloatTensor([[fx,    0,    w/2.0],
                                      [0.0,  fy,   h/2.0],
                                      [0.0, 0.0,  1.0]]).to(device)
        # k_matrix = torch.FloatTensor([[2*math.atan(0.5*w)/hfov, 0, img_w/2.0],
        #                               [0.0, 2*math.atan(0.5*h)/vfov, img_h/2.0],
        #                                 [0.0,    0.0,           1.0]]).to(device)
        k_inv = torch.inverse(k_matrix)
        k_inv = k_inv.view(1, 1, 1, 3, 3)
        k_matrix = k_matrix.view(1, 1, 1, 3, 3)
        # rays
        rays = torch.matmul(k_inv, hom_2d)
        if self.dataset == 'residential':
            # perform coordinate transformation
            R_swap = torch.FloatTensor([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
            # positive elevaltion around z
            R_z = torch.Tensor([[math.cos(elevation), -math.sin(elevation), 0],
                                [math.sin(elevation), math.cos(elevation), 0],
                                [0, 0, 1]])
            m_az = 2*math.pi - azimuth
            R_y = torch.Tensor([[math.cos(m_az), 0, math.sin(m_az)],
                                [0, 1, 0],
                                [-math.sin(m_az), 0, math.cos(m_az)]])
            R_mat = torch.matmul(R_y, torch.matmul(
                R_z, R_swap)).view(1, 1, 1, 3, 3)
        rotated_rays = torch.matmul(R_mat, rays).squeeze(-1)
        rotated_rays = rotated_rays / \
            torch.norm(rotated_rays, p=2, dim=-1,
                       keepdim=True).clamp(min=1e-08)
        # map cartesian to erp
        spherical_coords = self.cartesian_2_spherical(
            rotated_rays, normalized=True)
        uv_locations = self.spherical_2_equi(spherical_coords)
        sampled_image = self.sample_equi(input=input_imgs, grid=uv_locations)
        # return torch.flip(sampled_image, dims=(3,))
        return sampled_image

    def _get_unit_rays_on_sphere(self, device='cpu', b_size=1):
        x_y_locs = self.get_xy_coords(device=device).to(device)  # B, H, W, 2
        sph_locs = self.equi_2_spherical(x_y_locs).to(device)
        _, h, w, c = sph_locs.shape
        xyz_locs = self.spherical_2_cartesian(sph_locs).to(
            device).expand(b_size, h, w, c)  # B, H, W, 3
        return xyz_locs

    def get_vcoords(self, device='cpu'):
        erp = self.get_xy_coords(device=device)
        sph = self.equi_2_spherical(erp)
        _theta, phi, _rad = torch.split(
            sph, split_size_or_sections=1, dim=-1)
        v_coords = torch.sin(phi)
        return (v_coords)



if __name__ == '__main__':
    class Configs:
        pass
    configs = Configs()
    configs.__dict__['dataset'] = 'replica'
    configs.__dict__['batch_size'] = 2
    configs.__dict__['height'] = 256
    configs.__dict__['width'] = 512
    configs.__dict__['device'] = 'cpu'  # 'cuda:0'
    spt_utils = Utils(configs)
    thetas = torch.FloatTensor([1*math.pi/4.0, 0]).view(2)
    phis = torch.FloatTensor([1*math.pi/4.0, 0]).view(2)
    rot_mat = spt_utils.get_rotation_matrices(thetas, phis)
    # , trg_vec, src_vec
    # print(rot_mat.shape, trg_vec.shape, src_vec.shape)
    # trg_vec_est = torch.bmm(rot_mat, src_vec)
    features = torch.rand(configs.batch_size, 3, configs.height, configs.width)
    rot_feats = spt_utils.rotate_features(features, rot_mat)
    rot_mat_inv = torch.inverse(rot_mat)
    features_est = spt_utils.rotate_features(rot_feats, rot_mat_inv)
    print(torch.median(torch.abs(features_est - features)))
    print(torch.median(torch.abs(features_est)))
    # print(trg_vec[0])
    # print(trg_vec_est[0])
    # print(torch.norm(trg_vec_est[0], p=2))
