// RUN: XDSL_ROUNDTRIP

mesh.mesh @mesh0(shape = 2x2)
%0 = "test.op"() : () -> tensor<2x2xi8>
%1 = mesh.all_gather %0 on @mesh0 mesh_axes = [1] gather_axis = 1 : tensor<2x2xi8> -> tensor<2x4xi8>
%2 = mesh.all_reduce %0 on @mesh0 mesh_axes = [1, 0] reduction = max : tensor<2x2xi8> -> tensor<2x2xi8>
%3 = mesh.all_slice %0 on @mesh0 mesh_axes = [1] slice_axis = 1 : tensor<2x2xi8> -> tensor<2x1xi8>
%4 = mesh.all_to_all %0 on @mesh0 mesh_axes = [0] split_axis = 0 concat_axis = 0 : tensor<2x2xi8> -> tensor<2x2xi8>
%5 = mesh.broadcast %0 on @mesh0 mesh_axes = [0] root = [0] : (tensor<2x2xi8>) -> tensor<2x2xi8>
%6 = mesh.gather %0 on @mesh0 mesh_axes = [1] gather_axis = 1 root = [1] : (tensor<2x2xi8>) -> tensor<2x4xi8>
%7 = mesh.recv %0 on @mesh0 mesh_axes = [1] source = [1, 2] : (tensor<2x2xi8>) -> tensor<2x2xi8>
%8 = mesh.reduce %0 on @mesh0 mesh_axes = [1, 0] reduction = max root = [2, 3] : (tensor<2x2xi8>) -> tensor<2x2xi8>
%9 = mesh.reduce_scatter %0 on @mesh0 mesh_axes = [1] reduction = max scatter_axis = 0 : tensor<2x2xi8> -> tensor<1x2xi8>
%10 = mesh.scatter %0 on @mesh0 mesh_axes = [0] scatter_axis = 0 root = [1] : (tensor<2x2xi8>) -> tensor<1x2xi8>
%11 = mesh.send %0 on @mesh0 mesh_axes = [1] destination = [1, 2, 3] : (tensor<2x2xi8>) -> tensor<2x2xi8>
%12 = mesh.shift %0 on @mesh0 mesh_axes = [1] shift_axis = 1 offset = 2 rotate : tensor<2x2xi8> -> tensor<2x2xi8>


// CHECK:      mesh.mesh @mesh0(shape = 2x2)
// CHECK-NEXT: %0 = "test.op"() : () -> tensor<2x2xi8>
// CHECK-NEXT: %1 = mesh.all_gather %0 on @mesh0 mesh_axes = [1] gather_axis = 1 : tensor<2x2xi8> -> tensor<2x4xi8>
// CHECK-NEXT: %2 = mesh.all_reduce %0 on @mesh0 mesh_axes = [1, 0] reduction = max : tensor<2x2xi8> -> tensor<2x2xi8>
// CHECK-NEXT: %3 = mesh.all_slice %0 on @mesh0 mesh_axes = [1] slice_axis = 1 : tensor<2x2xi8> -> tensor<2x1xi8>
// CHECK-NEXT: %4 = mesh.all_to_all %0 on @mesh0 mesh_axes = [0] split_axis = 0 concat_axis = 0 : tensor<2x2xi8> -> tensor<2x2xi8>
// CHECK-NEXT: %5 = mesh.broadcast %0 on @mesh0 mesh_axes = [0] root = [0] : (tensor<2x2xi8>) -> tensor<2x2xi8>
// CHECK-NEXT: %6 = mesh.gather %0 on @mesh0 mesh_axes = [1] gather_axis = 1 root = [1] : (tensor<2x2xi8>) -> tensor<2x4xi8>
// CHECK-NEXT: %7 = mesh.recv %0 on @mesh0 mesh_axes = [1] source = [1, 2] : (tensor<2x2xi8>) -> tensor<2x2xi8>
// CHECK-NEXT: %8 = mesh.reduce %0 on @mesh0 mesh_axes = [1, 0] reduction = max root = [2, 3] : (tensor<2x2xi8>) -> tensor<2x2xi8>
// CHECK-NEXT: %9 = mesh.reduce_scatter %0 on @mesh0 mesh_axes = [1] reduction = max scatter_axis = 0 : tensor<2x2xi8> -> tensor<1x2xi8>
// CHECK-NEXT: %10 = mesh.scatter %0 on @mesh0 mesh_axes = [0] scatter_axis = 0 root = [1] : (tensor<2x2xi8>) -> tensor<1x2xi8>
// CHECK-NEXT: %11 = mesh.send %0 on @mesh0 mesh_axes = [1] destination = [1, 2, 3] : (tensor<2x2xi8>) -> tensor<2x2xi8>
// CHECK-NEXT: %12 = mesh.shift %0 on @mesh0 mesh_axes = [1] shift_axis = 1 offset = 2 rotate : tensor<2x2xi8> -> tensor<2x2xi8>
