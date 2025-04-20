// src/web/frontend/components/Layout.tsx
import React from 'react';
import { Box, Container, Heading, Text, VStack } from '@chakra-ui/react';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <Box minH="100vh" bg="gray.50">
      <Box bg="blue.600" color="white" py={6} mb={8}>
        <Container maxW="container.xl">
          <VStack spacing={2} align="flex-start">
            <Heading size="lg">Chest Cancer Classification</Heading>
            <Text>Upload a CT scan image for analysis</Text>
          </VStack>
        </Container>
      </Box>
      <Container maxW="container.xl" pb={8}>
        {children}
      </Container>
    </Box>
  );
};