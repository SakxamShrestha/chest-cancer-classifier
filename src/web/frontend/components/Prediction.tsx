// src/web/frontend/components/Prediction.tsx
import React from 'react';
import {
  Box,
  VStack,
  Text,
  Progress,
  Heading,
  Divider,
} from '@chakra-ui/react';
import { PredictionResponse } from '../types/api';

interface PredictionProps {
  prediction: PredictionResponse | null;
  isLoading: boolean;
}

export const Prediction: React.FC<PredictionProps> = ({ prediction, isLoading }) => {
  if (isLoading) {
    return (
      <Box p={4} borderRadius="md" bg="white" shadow="sm">
        <VStack spacing={4} align="stretch">
          <Text>Analyzing image...</Text>
          <Progress size="xs" isIndeterminate />
        </VStack>
      </Box>
    );
  }

  if (!prediction) return null;

  return (
    <Box p={6} borderRadius="lg" bg="white" shadow="md">
      <VStack spacing={4} align="stretch">
        <Heading size="md" color="blue.600">
          Prediction Results
        </Heading>
        
        <Box>
          <Text fontWeight="bold" mb={2}>
            Predicted Class:
          </Text>
          <Text fontSize="xl" color="green.600">
            {prediction.prediction}
          </Text>
        </Box>

        <Divider />

        <Box>
          <Text fontWeight="bold" mb={2}>
            Confidence Scores:
          </Text>
          {Object.entries(prediction.probabilities).map(([className, probability]) => (
            <Box key={className} mb={2}>
              <Text fontSize="sm" mb={1}>
                {className}
              </Text>
              <Box display="flex" alignItems="center">
                <Progress
                  value={probability * 100}
                  colorScheme={className === prediction.prediction ? "green" : "gray"}
                  width="100%"
                  mr={2}
                />
                <Text fontSize="sm" width="60px">
                  {(probability * 100).toFixed(1)}%
                </Text>
              </Box>
            </Box>
          ))}
        </Box>
      </VStack>
    </Box>
  );
};