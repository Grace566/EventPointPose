function [pose] = GetPoseLabels(XYZPOS, last_k, k, is_MeanLabel)
pose = zeros(13, 3);
if is_MeanLabel
    pose(1,:) = mean(XYZPOS.XYZPOS.head(last_k:k,:),1);
    pose(2,:) = mean(XYZPOS.XYZPOS.shoulderR(last_k:k,:),1);
    pose(3,:) = mean(XYZPOS.XYZPOS.shoulderL(last_k:k,:),1);
    pose(4,:) = mean(XYZPOS.XYZPOS.elbowR(last_k:k,:),1);
    pose(5,:) = mean(XYZPOS.XYZPOS.elbowL(last_k:k,:),1);
    pose(6,:) = mean(XYZPOS.XYZPOS.hipR(last_k:k,:),1);
    pose(7,:) = mean(XYZPOS.XYZPOS.hipL(last_k:k,:),1);
    pose(8,:) = mean(XYZPOS.XYZPOS.handR(last_k:k,:),1);
    pose(9,:) = mean(XYZPOS.XYZPOS.handL(last_k:k,:),1);
    pose(10,:) = mean(XYZPOS.XYZPOS.kneeR(last_k:k,:),1);
    pose(11,:) = mean(XYZPOS.XYZPOS.kneeL(last_k:k,:),1);
    pose(12,:) = mean(XYZPOS.XYZPOS.footR(last_k:k,:),1);
    pose(13,:) = mean(XYZPOS.XYZPOS.footL(last_k:k,:),1);
else
    pose(1,:) = XYZPOS.XYZPOS.head(k,:);
    pose(2,:) = XYZPOS.XYZPOS.shoulderR(k,:);
    pose(3,:) = XYZPOS.XYZPOS.shoulderL(k,:);
    pose(4,:) = XYZPOS.XYZPOS.elbowR(k,:);
    pose(5,:) = XYZPOS.XYZPOS.elbowL(k,:);
    pose(6,:) = XYZPOS.XYZPOS.hipR(k,:);
    pose(7,:) = XYZPOS.XYZPOS.hipL(k,:);
    pose(8,:) = XYZPOS.XYZPOS.handR(k,:);
    pose(9,:) = XYZPOS.XYZPOS.handL(k,:);
    pose(10,:) = XYZPOS.XYZPOS.kneeR(k,:);
    pose(11,:) = XYZPOS.XYZPOS.kneeL(k,:);
    pose(12,:) = XYZPOS.XYZPOS.footR(k,:);
    pose(13,:) = XYZPOS.XYZPOS.footL(k,:);
end