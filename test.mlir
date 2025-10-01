mesh.mesh @grid0(shape = 4)

%a = "test.op"() : () -> tensor<1024xf32>
%b = "test.op"() : () -> tensor<1024xf32>

// define the sharding: split the first and only tensor dimension across axis 0 of grid0
%sharding = mesh.sharding @grid0 split_axes = [[0]] : !mesh.sharding

// annotate as, bs with how they should be sharded
// they should be sharded across @grid0 along the
// splitting the tensors' 0th dimension (0th subarray)
// 1024 across the 0th axis ([0]) of the grid.
%as = mesh.shard %a to %sharding : tensor<1024xf32>
%bs = mesh.shard %b to %sharding : tensor<1024xf32>

// perform computation: addf or tosa here?? pros and cons??
// tosa may mean we don't have to insert an addf when we see
// tosa.add but otherwise not sure of differences: small issue only not important
%cs = arith.addf %as, %bs : tensor<1024xf32>

// Gather shard on all devices?
%c = mesh.all_gather %cs on @grid0 mesh_axes = [0] gather_axis = 0 : tensor<256xf32> -> tensor<1024xf32>
