// src/web/frontend/components/Prediction.tsx
import React from 'react';
import {
  Box,
  Text,
  VStack,
  Progress,
  Badge,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  SimpleGrid,
} from '@chakra-ui/react';
import { PredictionResponse } from '../types/api';

interface PredictionProps {
  prediction: PredictionResponse | null;
  isLoading: boolean;
}

export const Prediction: React.FC<PredictionProps> = ({ prediction, isLoading }) => {
  const getBadgeColor = (className: string | undefined) => {
    if (!className) return 'gray';
    
    const normalizedClassName = className.toLowerCase();
    switch (normalizedClassName) {
      case 'adenocarcinoma':
        return 'red';
      case 'large.cell.carcinoma':
        return 'orange';
      case 'squamous.cell.carcinoma':
        return 'purple';
      case 'normal':
        return 'green';
      default:
        return 'gray';
    }
  };

  if (isLoading) {
    return (
      <Box p={6} bg="white" borderRadius="xl" borderWidth="1px" borderColor="gray.200">
        <VStack spacing={4} align="stretch">
          <Text fontSize="lg" fontWeight="medium">Analyzing Image...</Text>
          <Progress size="sm" isIndeterminate colorScheme="blue" />
        </VStack>
      </Box>
    );
  }

  if (!prediction) {
    return (
      <Box p={6} bg="white" borderRadius="xl" borderWidth="1px" borderColor="gray.200">
        <Text color="gray.500">Upload an image to see the prediction</Text>
      </Box>
    );
  }

  // Get the highest probability class
  const predictedClass = prediction.prediction;
  const confidence = prediction.probabilities[predictedClass] || 0;

  return (
    <Box p={6} bg="white" borderRadius="xl" borderWidth="1px" borderColor="gray.200">
      <VStack spacing={6} align="stretch">
        <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
          <Stat>
            <StatLabel fontSize="lg">Classification Result</StatLabel>
            <StatNumber>
              <Badge
                colorScheme={getBadgeColor(predictedClass)}
                fontSize="md"
                p={2}
                borderRadius="md"
              >
                {predictedClass ? predictedClass.replace(/\./g, ' ') : 'Unknown'}
              </Badge>
            </StatNumber>
            <StatHelpText>
              Based on CT scan analysis
            </StatHelpText>
          </Stat>
          <Stat>
            <StatLabel fontSize="lg">Confidence Level</StatLabel>
            <StatNumber>{(confidence * 100).toFixed(2)}%</StatNumber>
            <Progress
              value={confidence * 100}
              colorScheme={confidence > 0.7 ? 'green' : 'orange'}
              size="sm"
              borderRadius="full"
            />
          </Stat>
        </SimpleGrid>
      </VStack>
    </Box>
  );
};