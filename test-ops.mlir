mesh.mesh @mesh0(shape = 2x2)
%0 = "test.op"() : () -> tensor<2x2xi8>
%1 = mesh.all_gather %0 on @mesh0 mesh_axes = [1] gather_axis = 1 : tensor<2x2xi8> -> tensor<2x4xi8>
%2 = mesh.all_reduce %0 on @mesh0 mesh_axes = [1, 0] reduction = max : tensor<2x2xi8> -> tensor<2x2xi8>
%3 = mesh.all_slice %0 on @mesh0 mesh_axes = [1] slice_axis = 1 : tensor<2x2xi8> -> tensor<2x1xi8>
%4 = mesh.all_to_all %0 on @mesh0 mesh_axes = [0] split_axis = 0 concat_axis = 0 : tensor<2x2xi8> -> tensor<2x2xi8>
%5 = mesh.broadcast %0 on @mesh0 mesh_axes = [0] root = [0] : (tensor<2x2xi8>) -> tensor<2x2xi8>
%6 = mesh.gather %0 on @mesh0 mesh_axes = [1] gather_axis = 1 root = [1] : (tensor<2x2xi8>) -> tensor<2x4xi8>
