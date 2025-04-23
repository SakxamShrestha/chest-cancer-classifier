// src/web/frontend/components/Layout.tsx
import React from 'react';
import {
  Box,
  Container,
  Heading,
  Text,
  VStack,
  Flex,
  Icon,
} from '@chakra-ui/react';
import { FiActivity } from 'react-icons/fi';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <Box minH="100vh" bg="gray.50">
      <Box 
        as="header" 
        py={4} 
        bg="white" 
        boxShadow="sm"
        position="sticky"
        top={0}
        zIndex={10}
      >
        <Container maxW="container.xl">
          <Flex align="center" gap={3}>
            <Icon as={FiActivity} w={8} h={8} color="blue.500" />
            <VStack align="flex-start" spacing={0}>
              <Heading size="md">Chest Cancer Classifier</Heading>
              <Text fontSize="sm" color="gray.500">AI-Powered Medical Analysis</Text>
            </VStack>
          </Flex>
        </Container>
      </Box>
      <Container maxW="container.xl" py={8}>
        {children}
      </Container>
      <Box as="footer" py={6} textAlign="center">
        <Text color="gray.500">
          © {new Date().getFullYear()} Chest Cancer Classifier. All rights reserved.
        </Text>
      </Box>
    </Box>
  );
};