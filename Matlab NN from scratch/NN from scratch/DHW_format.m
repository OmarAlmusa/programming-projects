function output = DHW_format(img)
    s = size(img);
    cas = zeros(s(3), s(1), s(2));
    for depth=1:s(3)
        cas(depth, :, :) = reshape(img(:, :, depth), 1, s(1), s(2));
    end
    output = cas;
end

