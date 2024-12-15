import torch
import torch.nn as nn

class SiameseExEy(nn.Module):
    def __init__(self):
        super(SiameseExEy, self).__init__()
        self.feature_extraction = FeatureExtraction()
        self.dense_layer_ex_ey = nn.Sequential(
            nn.Linear(256*2*2*2, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(512, 256),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(256,256),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(256,128),
            nn.InstanceNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(128,2)
        )

    def forward(self, image1, image2):
        if image2.dim() == 3:
            image2 = image2.expand(image1.size(0), -1, -1, -1)

        features1 = self.feature_extraction(image1)
        features2 = self.feature_extraction(image2)


        features1 = features1.view(features1.size(0), -1)
        features2 = features2.view(features2.size(0), -1)

        combined_features = torch.cat((features1, features2), dim=1)
        return self.dense_layer_ex_ey(combined_features)

class SiameseEz(nn.Module):
    def __init__(self):
        super(SiameseEz, self).__init__()
        self.feature_extraction = FeatureExtraction()

        self.dense_layer_ez = nn.Sequential(
            nn.Linear(256*2*2*2, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(512, 256),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(256,256),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(256,128),
            nn.InstanceNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(128,1)
        )

    def forward(self, image1, image2):
        if image2.dim() == 3:
            image2 = image2.expand(image1.size(0), -1, -1, -1)

        features1 = self.feature_extraction(image1)
        features2 = self.feature_extraction(image2)

        features1 = features1.view(features1.size(0), -1)
        features2 = features2.view(features2.size(0), -1)

        combined_features = torch.cat((features1, features2), dim=1)
        return self.dense_layer_ez(combined_features)

class SiameseEr(nn.Module):
    def __init__(self):
        super(SiameseEr, self).__init__()
        self.feature_extraction = FeatureExtraction()
        self.dense_layer_er = nn.Sequential(
            nn.Linear(256*2*2*2, 512),
            nn.InstanceNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(512, 256),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(256,256),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(256,128),
            nn.InstanceNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(128,2)
        )

    def forward(self, image1, image2):
        if image2.dim() == 3:
            image2 = image2.expand(image1.size(0), -1, -1, -1)

        features1 = self.feature_extraction(image1)
        features2 = self.feature_extraction(image2)

        features1 = features1.view(features1.size(0), -1)
        features2 = features2.view(features2.size(0), -1)

        combined_features = torch.cat((features1, features2), dim=1)
        return self.dense_layer_er(combined_features)

class FeatureExtraction(nn.Module):

    def __init__(self):
        super(FeatureExtraction, self).__init__()

        self.feature_extraction_cnn = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3,stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Conv2d(256, 256, kernel_size=3,stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Conv2d(256, 256, kernel_size=3,stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Conv2d(256, 256, kernel_size=3,stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Conv2d(256, 256, kernel_size=3,stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )
    def forward(self, x):
        return self.feature_extraction_cnn(x)